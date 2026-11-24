#!/usr/bin/env python3
"""
vLLM LoRA Inference Script for SLURM clusters (Ray Distributed).

Supports:
    - Single-node and multi-node inference
    - Tensor parallelism
    - Pipeline parallelism
    - Base model inference
    - LoRA inference
    - Quantized model inference

The Ray cluster must be initialized before running this script.
The script is executed only once on the Ray head node.

Reference:
https://docs.vllm.ai/en/v0.12.0/serving/parallelism_scaling/
Adapted from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_example.py
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import ray

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args():

    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run Distributed vLLM inference with LoRA using Ray'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to base model directory'
    )

    parser.add_argument(
        '--lora-path',
        type=str,
        required=True,
        help='Path to LoRA adapter directory'
    )

    parser.add_argument(
        '--prompt-file',
        type=str,
        default='prompt_QA.txt',
        help='Path to file containing prompts (.txt or .json)'
    )

    parser.add_argument(
        '--quantization',
        type=str,
        default=os.environ.get("QUANTIZATION", "None"),
        choices=[
            "None",
            "bitsandbytes",
            "awq",
            "gptq",
            "fp8"
        ],
        help='Quantization method [default: None]'
    )

    parser.add_argument(
        '--max-lora-rank',
        type=int,
        default=int(os.environ.get("MAX_LORA_RANK", "64")),
        help='Maximum LoRA rank [default: 64]'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature [default: 0.0]'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='Maximum tokens to generate [default: 128]'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Optional: Save outputs to file'
    )

    # Distributed specific arguments - can be set from SLURM

    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=4,
        help='Number of GPUs per tensor parallel group [default: 4]'
    )

    parser.add_argument(
        '--pipeline-parallel-size',
        type=int,
        default=1,
        help='Number of pipeline parallel stages [default: 1]'
    )

    return parser.parse_args()


def load_prompts_from_file(file_path: str) -> list[str]:

    """Load prompts from a text or JSON file."""

    path = Path(file_path)

    if not path.exists():

        print(
            f"ERROR: Prompt file not found: {file_path}",
            file=sys.stderr
        )

        sys.exit(1)

    prompts = []

    if path.suffix.lower() == '.json':

        with open(path, 'r', encoding='utf-8') as f:

            try:

                prompts = json.load(f)

                if not isinstance(prompts, list) or not all(
                    isinstance(p, str) for p in prompts
                ):

                    print(
                        "ERROR: JSON file must contain a list of strings.",
                        file=sys.stderr
                    )

                    sys.exit(1)

            except json.JSONDecodeError as e:

                print(
                    f"ERROR: Failed to parse JSON in {file_path}: {e}",
                    file=sys.stderr
                )

                sys.exit(1)

    else:

        with open(path, 'r', encoding='utf-8') as f:

            for line in f:

                prompt = line.strip()

                if prompt and not prompt.startswith('#'):

                    prompts.append(prompt)

    if not prompts:

        print(
            f"ERROR: No valid prompts found in {file_path}",
            file=sys.stderr
        )

        sys.exit(1)

    print(f"Loaded {len(prompts)} prompt(s) from {file_path}")

    for i, p in enumerate(prompts, 1):

        preview_text = p.replace('\n', ' ')

        print(
            f"  [{i}] {preview_text[:80]}"
            f"{'...' if len(preview_text) > 80 else ''}"
        )

    return prompts


def initialize_ray(args):

    """Connect to the existing Ray cluster and verify resources."""

    print("\n" + "=" * 60)

    print("Initializing Ray Distributed Environment")

    print("=" * 60)

    # Connect to the Ray cluster initialized by the SLURM script

    ray_address = os.environ.get("RAY_ADDRESS", "auto")

    print(f"Ray address: {ray_address}")

    ray.init(
        address=ray_address,
        ignore_reinit_error=True
    )

    # Retrieve cluster resources

    cluster_resources = ray.cluster_resources()

    available_resources = ray.available_resources()

    alive_nodes = [
        node for node in ray.nodes()
        if node["Alive"]
    ]

    total_gpus = int(cluster_resources.get("GPU", 0))

    available_gpus = int(available_resources.get("GPU", 0))

    total_cpus = int(cluster_resources.get("CPU", 0))

    required_gpus = (
        args.tensor_parallel_size *
        args.pipeline_parallel_size
    )

    print(f"\nNumber of Ray nodes: {len(alive_nodes)}")

    print(f"Total CPUs: {total_cpus}")

    print(f"Total GPUs: {total_gpus}")

    print(f"Available GPUs: {available_gpus}")

    print(f"Required GPUs: {required_gpus}")

    print("\nRay node information:")

    for i, node in enumerate(alive_nodes):

        node_resources = node.get("Resources", {})

        print(
            f"  Node {i}: "
            f"IP={node['NodeManagerAddress']}, "
            f"CPUs={int(node_resources.get('CPU', 0))}, "
            f"GPUs={int(node_resources.get('GPU', 0))}"
        )

    # Verify that Ray has enough GPU resources

    if total_gpus < required_gpus:

        raise RuntimeError(
            f"Insufficient GPU resources in the Ray cluster. "
            f"Required: {required_gpus}, "
            f"Registered: {total_gpus}"
        )

    if available_gpus < required_gpus:

        raise RuntimeError(
            f"Insufficient available GPU resources. "
            f"Required: {required_gpus}, "
            f"Available: {available_gpus}"
        )

    print("\nRay cluster is ready!")

    print("=" * 60)


def main():

    """Main function."""

    args = parse_args()

    # Verify parallelism settings

    if args.tensor_parallel_size < 1:

        raise ValueError("Tensor parallel size must be >= 1")

    if args.pipeline_parallel_size < 1:

        raise ValueError("Pipeline parallel size must be >= 1")

    # With Ray, this script is executed only once on the head node.
    # Ray and vLLM manage the distributed GPU worker processes.

    quant_display = (
        "None (Model default dtype)"
        if args.quantization == "None"
        else args.quantization
    )

    print(f"""
{'=' * 60}

vLLM Distributed LoRA Inference (Ray)

{'=' * 60}

Model: {args.model}

LoRA: {args.lora_path}

Prompts: {args.prompt_file}

Quantization: {quant_display}

Max LoRA Rank: {args.max_lora_rank}

TP Size: {args.tensor_parallel_size}

PP Size: {args.pipeline_parallel_size}

Total GPUs: {args.tensor_parallel_size * args.pipeline_parallel_size}

{'=' * 60}
""")

    # Load prompts

    prompts = load_prompts_from_file(args.prompt_file)

    # Configure sampling parameters

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens
    )

    llm = None

    try:

        # --- Initialize Ray Distributed Environment ---

        initialize_ray(args)

        print("\nInitializing vLLM engine ...")

        # Configure the LLM Arguments

        quant_value = (
            None
            if args.quantization == "None"
            else args.quantization
        )

        llm = LLM(

            model=args.model,

            quantization=quant_value,

            enable_lora=True,

            max_lora_rank=args.max_lora_rank,

            max_loras=4,

            tensor_parallel_size=args.tensor_parallel_size,

            pipeline_parallel_size=args.pipeline_parallel_size,

            # Ray manages distributed GPU worker processes

            distributed_executor_backend="ray",

            # Set random seed for reproducibility

            seed=42
        )

        # Apply the model-specific chat template

        tokenizer = llm.get_tokenizer()

        formatted_prompts = []

        for p in prompts:

            message = [
                {
                    "role": "user",
                    "content": p
                }
            ]

            formatted_prompts.append(

                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )

            )

        results = []


        # =============================================================
        # --- Run Base Inference ---
        # =============================================================

        print("\n" + "-" * 60)

        print("Running Base Inference...")

        print("-" * 60)

        base_outputs = llm.generate(
            formatted_prompts,
            sampling_params
        )

        for i, output in enumerate(base_outputs):

            clean_prompt = prompts[i]

            print(f"[Base] Prompt {i+1}:\n{clean_prompt}")

            print(f"Output: {output.outputs[0].text}\n")

            results.append({

                'request_id': f"base_{i}",

                'prompt': clean_prompt,

                'output': output.outputs[0].text,

                'type': "Base"

            })


        # =============================================================
        # --- Run LoRA Inference ---
        # =============================================================

        print("-" * 60)

        print("Running LoRA Inference...")

        print("-" * 60)

        lora_req = LoRARequest(
            "lora-test",
            1,
            args.lora_path
        )

        lora_outputs = llm.generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_req
        )

        for i, output in enumerate(lora_outputs):

            clean_prompt = prompts[i]

            print(f"[LoRA] Prompt {i+1}:\n{clean_prompt}")

            print(f"Output: {output.outputs[0].text}\n")

            results.append({

                'request_id': f"lora_{i}",

                'prompt': clean_prompt,

                'output': output.outputs[0].text,

                'type': "LoRA"

            })


        # =============================================================
        # --- Save Results ---
        # =============================================================

        if args.output_file:

            output_path = Path(args.output_file)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(output_path, 'w', encoding='utf-8') as f:

                json.dump(
                    results,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(f"\nResults saved to: {args.output_file}")

        print("\n" + "~" * 60)

        print("ALL INFERENCE COMPLETED SUCCESSFULLY")

        print("~" * 60)


    # =================================================================
    # --- Cleanup ---
    # =================================================================

    finally:

        print("\nCleaning up vLLM and Ray resources ...")

        if llm is not None:

            del llm

        gc.collect()

        # Disconnect the Python driver from the existing Ray cluster.
        # The SLURM script is responsible for stopping the Ray cluster.

        if ray.is_initialized():

            ray.shutdown()

        print("Disconnected from Ray cluster.")


if __name__ == '__main__':

    main()
