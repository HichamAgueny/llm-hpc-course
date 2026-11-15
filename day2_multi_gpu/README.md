# Day 2: Distributed Training, Quantization, Serving & Inference

This directory contains labs, configurations, and job scripts for scaling LLM workloads across multiple GPUs and multiple nodes on the Olivia supercomputer.

## Learning Objectives

- Scale fine-tuning across multiple GPUs (single-node) and multiple nodes using PyTorch FSDP.
- Compare **LoRA**, **QLoRA**, and **QAT-LoRA** (Quantization-Aware Training) for Question-Answering (QA) tasks.
- Profile distributed workloads to analyze inter-GPU communication and memory scaling.
- Apply post-training quantization techniques using **BitsAndBytes**, **TorchAO**, and **GPTQModel**.
- Deploy high-throughput LLM serving and inference pipelines using **vLLM** (single-device and multi-device tensor parallel).

---

## Before You Start

From the course folder (where `run.sh` is):
```bash
source course_env.sh    # once per login: sets the Slurm account & reservation and $COURSE_ROOT
```
(If this is your first session, run `./setup_env.sh` first — see the [main README](../README.md).)

**You never need to edit paths or accounts in the job scripts.** Every lab can be started with the
**fast launcher** `./run.sh day2 ...` from the course folder (or `./menu.sh`), or by **direct Slurm
submission** with the commands below, which use `$COURSE_ROOT` and therefore work from any folder.
Run `./run.sh help` to see all options.

---

## Directory Contents

| Folder | Description |
| :--- | :--- |
| `fine-tuning/` | Slurm job scripts for multi-GPU single-node (`ft_single_node/`) and multi-node (`ft_multi_node/`) training (LoRA, QLoRA, QAT-LoRA) via Apptainer and EESSI. |
| `evaluation/` | Multi-GPU evaluation scripts (MMLU benchmark) with LoRA adapter merging and base model baseline. |
| `profiling/` | Multi-GPU distributed profiling scripts for measuring communication overhead and kernel efficiency. |
| `quantization/` | Post-training quantization labs: BitsAndBytes (4-bit & 8-bit), TorchAO, and GPTQModel. |
| `serving/` | vLLM OpenAI-compatible server deployment (single-device and multi-device, with container or EESSI) and interactive `chat.sh` clients. |
| `inference/` | Distributed batch inference scripts powered by vLLM with LoRA adapter support. |

---

## Where Things Are Written

- **Slurm log of each job:** the `out/` folder next to the job script.
- **Results:** under `results/` in the course folder:
  - **`results/checkpoint_results/`**: Checkpoints and adapter weights generated during fine-tuning (e.g., `llama3.2-1b_lora_multi_device/checkpoints/`). EESSI runs use the prefix `eessi_`.
  - **`results/profiling_results/`**: Distributed profiling outputs and logs.
  - **`results/evaluation_results/`**: MMLU benchmark output files and summary metrics.
  - **`results/quantized_models/`**: Your quantized models.
  - **`results/benchmark_logs/`**: vLLM serving benchmark results.

---

## Guided Hands-On Lab Worksheets (Day 2)

### **Lab 2.1: Multi-GPU Distributed Fine-Tuning (Single Node)**
* **Objective:** Scale fine-tuning across 4 GPUs on a single node using PyTorch Fully Sharded Data Parallel (FSDP).
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day2 ft-single --method=lora              # LoRA (container)
    ./run.sh day2 ft-single --method=lora --env=eessi  # LoRA (EESSI)
    ./run.sh day2 ft-single --method=qlora             # QLoRA (container only)
    ./run.sh day2 ft-single --method=qat-lora          # QAT-LoRA (container only)
    ```
  * Direct Slurm submission:
    ```bash
    # LoRA (Container):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_container && sbatch job_multiGPU_QA_LoRA.sh
    # LoRA (EESSI):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_eessi && sbatch job_multiGPU_QA_LoRA_eessi.sh
    # QLoRA (4 GPUs):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_single_node/qlora_slurm && sbatch job_multiGPU_QA_QLoRA.sh
    # QAT-LoRA (4 GPUs):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_single_node/qat-lora_slurm && sbatch job_multiGPU_QA_qat-LoRA.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `out/` in the folder of the script you submitted (e.g. `ft_single_node/lora_slurm/job_container/out/`, `ft_single_node/qlora_slurm/out/`)
  * **Training Loss & Metrics:** `results/checkpoint_results/llama3.2-1b_<method>_multi_device/logs/`
  * **Checkpoints:** `results/checkpoint_results/llama3.2-1b_<method>_multi_device/checkpoints/`
* **Student Focus:** Use `./run.sh monitor <JobID>` to observe balanced workload distribution and memory utilization across all 4 GPUs simultaneously.

---

### **Lab 2.2: Multi-Node Distributed Scaling**
* **Objective:** Scale training across multiple cluster nodes with optimized Slingshot interconnect and AWS-OFI-NCCL network bindings.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day2 ft-multi --method=lora               # Multi-node LoRA (container)
    ./run.sh day2 ft-multi --method=lora --env=eessi   # Multi-node LoRA (EESSI)
    ./run.sh day2 ft-multi --method=qlora              # Multi-node QLoRA (container only)
    ./run.sh day2 ft-multi --method=qat-lora           # Multi-node QAT-LoRA (container only)
    ```
  * Direct Slurm submission:
    ```bash
    # LoRA (Container):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_container && sbatch job_multiNode_QA_LoRA.sh
    # LoRA (EESSI):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_eessi && sbatch job_multiNode_QA_LoRA_eessi.sh
    # QLoRA (Multi-Node):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm && sbatch job_multiNode_QA_QLoRA.sh
    # QAT-LoRA (Multi-Node):
    cd $COURSE_ROOT/day2_multi_gpu/fine-tuning/ft_multi_node/qat-lora_slurm && sbatch job_multiNode_QA_qat-LoRA.sh
    ```
* **Network settings:** all multi-node jobs load the shared Slingshot/NCCL settings from [`utils/nccl_env.sh`](../utils/nccl_env.sh).
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `out/` in the folder of the script you submitted
  * **Checkpoints:** `results/checkpoint_results/llama3.2-1b_<method>_multi_node/checkpoints/`

---

### **Lab 2.3: Distributed Profiling & Communication Bottleneck Analysis**
* **Objective:** Profile multi-GPU training to identify inter-GPU collective communication overhead in FSDP.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day2 profile --method=lora
    ./run.sh day2 profile --method=qlora
    ./run.sh day2 profile --method=qat-lora
    ```
  * Direct Slurm submission:
    ```bash
    cd $COURSE_ROOT/day2_multi_gpu/profiling
    sbatch job_multiGPU_QA_LoRA.sh       # Profile LoRA
    sbatch job_multiGPU_QA_QLoRA.sh      # Profile QLoRA
    sbatch job_multiGPU_QA_qat-LoRA.sh   # Profile QAT-LoRA
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/profiling/out/*.out`
  * **Per-Rank Traces:** `results/profiling_results/profiling_llama3.2-1b_<method>_multi_device/profiles/` (`rank0_...`, `rank1_...`, etc.)
* **Viewing Distributed Traces Locally** (on your laptop, shown for the recommended clone location):
  ```bash
  scp -r USERNAME@olivia.sigma2.no:/cluster/work/projects/nn9971k/USERNAME/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_multi_device/profiles .
  open profiles/rank0_memory-timeline.html
  ```
* **Key Analytical Questions for Students:**
  * What proportion of iteration time is spent in NCCL collective calls (`all_reduce`, `reduce_scatter`) vs. local matrix multiplication compute?
  * How does NVLink intra-node communication efficiency compare against single-GPU kernel execution?

---

### **Lab 2.4: Multi-GPU Distributed Evaluation**
* **Objective:** Evaluate multi-GPU fine-tuned checkpoints on the MMLU benchmark across 4 GPUs.
* **Checkpoint Selection:** By default the Day 1 LoRA adapter is evaluated
  (`results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0`). To evaluate your
  Day 2 adapter instead:
  ```bash
  LORA_PATH=$COURSE_ROOT/results/checkpoint_results/llama3.2-1b_lora_multi_device/checkpoints/epoch_0 ./run.sh day2 eval
  ```
  The merged model is saved next to the adapter (`<adapter>_merged`), so each adapter gets its own merge.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day2 eval                 # Evaluate LoRA adapter (merges adapter first)
    ./run.sh day2 eval --base          # Evaluate base model baseline
    ./run.sh day2 eval --env=eessi     # Same with EESSI modules (add --base for the baseline)
    ```
  * Direct Slurm submission:
    ```bash
    cd $COURSE_ROOT/day2_multi_gpu/evaluation && sbatch job_multiGPU_LoRA.sh         # LoRA (container)
    cd $COURSE_ROOT/day2_multi_gpu/evaluation && sbatch job_multiGPU_No_LoRA.sh      # base model (container)
    cd $COURSE_ROOT/day2_multi_gpu/evaluation && sbatch job_multiGPU_LoRA_eessi.sh   # LoRA (EESSI)
    cd $COURSE_ROOT/day2_multi_gpu/evaluation && sbatch job_multiGPU_No_LoRA_eessi.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/evaluation/out/*.out`
  * **Benchmark Metrics:** `results/evaluation_results/llama3.2-1b-mmlu_multigpu/` (base: `llama3.2-1b-base-mmlu/`; EESSI: `llama3.2-1b-mmlu_eessi/`, `llama3.2-1b-base-mmlu_eessi/`)

---

### **Lab 2.5: Post-Training Quantization (PTQ)**
* **Objective:** Quantize base model weights using diverse quantization backends without retraining.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day2 quantize bnb4      # BitsAndBytes 4-bit
    ./run.sh day2 quantize bnb8      # BitsAndBytes 8-bit
    ./run.sh day2 quantize torchao   # TorchAO INT8 weight-only
    ./run.sh day2 quantize gptq      # GPTQModel 4-bit
    ```
  * Direct Slurm submission:
    ```bash
    cd $COURSE_ROOT/day2_multi_gpu/quantization/quat_bitsandbytes && sbatch job_bnb_4bit.sh
    cd $COURSE_ROOT/day2_multi_gpu/quantization/quat_bitsandbytes && sbatch job_bnb_8bit.sh
    cd $COURSE_ROOT/day2_multi_gpu/quantization/quat_torchao && sbatch job_torchao.sh
    cd $COURSE_ROOT/day2_multi_gpu/quantization/quat_gptqmodel && sbatch job_gptq.sh
    cd $COURSE_ROOT/day2_multi_gpu/quantization/quat_gptqmodel && sbatch job_gptq_with_filter.sh   # GPTQ, filtered calibration data
    ```
  GPTQ uses the calibration file `quat_gptqmodel/calib_english.txt`; if it is missing, the job downloads it first.
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/quantization/<backend>/out/*.out`
  * **Saved Quantized Model:** `results/quantized_models/Llama-3.2-1B-Instruct-<backend>` (your own copy; the serving lab uses it automatically when it exists, otherwise it falls back to the shared pre-quantized model)
* **Student Focus:** Inspect file sizes of the generated model weights and compare the on-disk footprint across FP16, INT8, and INT4. The end of each job log prints the `du -sh` command to do this.

---

### **Lab 2.6: High-Throughput Serving & Interactive Terminal Chat with vLLM**
* **Objective:** Deploy an OpenAI-compatible vLLM serving engine with dynamic LoRA adapter loading and connect using an interactive terminal chat interface.
* **Choose a server** (options can be combined):

  | Option | Values | Default |
  | :--- | :--- | :--- |
  | `--type` | `single` (1 GPU) or `multi` (4 GPUs, tensor parallel) | `multi` |
  | `--env` | `container` or `eessi` | `container` |
  | `--quant` | `none`, `bnb`, `gptq`, `torchao` | `none` |
  | `LORA_PATH=` | adapter folder to load, or `""` for no adapter | Day 1 LoRA adapter |
  | `PORT=` | server port (pick another one if 8000 is busy on the node) | `8000` |

* **Steps:**
  1. Start the server (it runs as a Slurm job for up to 30 minutes):
     ```bash
     ./run.sh day2 serve                                   # 4 GPUs, container, Day 1 LoRA adapter
     ./run.sh day2 serve --type=single --quant=bnb         # 1 GPU, BitsAndBytes model + adapter
     LORA_PATH="" ./run.sh day2 serve --env=eessi          # base model only, EESSI
     # or direct:
     cd $COURSE_ROOT/day2_multi_gpu/serving/multi_device/vllm_container && sbatch serve_vllm.sh
     ```
  2. Wait until the server is ready (the log shows `Application startup complete`):
     ```bash
     tail -f $COURSE_ROOT/day2_multi_gpu/serving/multi_device/vllm_container/out/*.out
     ```
  3. Chat with it — use the **same `--type` and `--env`** as for `serve`:
     ```bash
     ./run.sh day2 chat                                    # matches the default server
     ./run.sh day2 chat --type=single                      # matches a --type=single server
     ```
     Type `exit` to leave the chat.
  4. Optional — benchmark the running single-GPU container server:
     ```bash
     ./recipes/bench/run_benchmark.sh      # results in results/benchmark_logs/
     ```
  5. Stop the server when you are done: `scancel <JobID>`.
* **Interactive alternative:** start a GPU shell with `bash srun_sinteractive` (1 GPU) or
  `bash srun_sinteractive_4gpus` (4 GPUs) inside a serving folder, then run the server script directly,
  e.g. `./serve_vllm.sh`. Chat from a second login terminal with `./chat.sh` in the same folder.
* **Student Focus:** Interact with your fine-tuned QA model directly from the command line and observe how vLLM handles token streaming.

---

### **Lab 2.7: Distributed High-Throughput Batch Inference**
* **Objective:** Benchmark batch inference throughput using vLLM tensor parallelism across multiple GPUs.
* **Launch Commands:**
  * Fast launcher (from the course folder):
    ```bash
    ./run.sh day2 inference               # 4 GPUs, container
    ./run.sh day2 inference --env=eessi   # 4 GPUs, EESSI
    ```
  * Direct Slurm submission:
    ```bash
    cd $COURSE_ROOT/day2_multi_gpu/inference && sbatch slurm_vllm_distributed_container.sh
    cd $COURSE_ROOT/day2_multi_gpu/inference && sbatch slurm_vllm_distributed_eessi.sh
    cd $COURSE_ROOT/day2_multi_gpu/inference && sbatch slurm_vllm_lora_singlegpu.sh   # 1-GPU comparison
    ```
  The prompts come from `inference/prompt_QA.json`; choose another adapter with `LORA_PATH=...` as in Lab 2.4.
* **Where to Inspect Outputs:**
  * **Throughput & Latency Metrics:** `day2_multi_gpu/inference/out/*.out`
* **Student Focus:** Inspect the generated tokens/second throughput and per-token generation latency reported in the Slurm log.

---

## If a Job Fails

1. Open the newest log (inside the lab folder): `tail -50 $(ls -t out/*.out | head -1)`
2. Messages starting with `ERROR: ... not found` say exactly what is missing and how to fix it
   (for example: fine-tune first, or run the quantization lab).
3. `sbatch: error: ... account` → run `source course_env.sh` and submit again (or use `./run.sh`).
4. `./check_env.sh` re-checks models, datasets, container images and folders.
