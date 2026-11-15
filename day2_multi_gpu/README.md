# Day 2: Distributed Training, Quantization, Serving & Inference

This directory contains labs, configurations, and job scripts for scaling LLM workloads across multiple GPUs and multiple nodes on the Olivia supercomputer.

## Learning Objectives

- Scale fine-tuning across multiple GPUs (single-node) and multiple nodes using PyTorch FSDP.
- Compare **LoRA**, **QLoRA**, and **QAT-LoRA** (Quantization-Aware Training) for Question-Answering (QA) tasks.
- Profile distributed workloads to analyze inter-GPU communication and memory scaling.
- Apply post-training quantization techniques using **BitsAndBytes**, **TorchAO**, and **GPTQModel**.
- Deploy high-throughput LLM serving and inference pipelines using **vLLM** (single-device and multi-device tensor parallel).

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

## Output Directory Organization

Outputs conform to the course-wide `results/` folder hierarchy:

- **`${PROJECT_DIR}/results/checkpoint_results/`**: Checkpoints and adapter weights generated during fine-tuning (e.g., `llama3.2-1b_lora_multi_device/checkpoints/`).
- **`${PROJECT_DIR}/results/profiling_results/`**: Distributed profiling outputs and logs.
- **`${PROJECT_DIR}/results/evaluation_results/`**: MMLU benchmark output files and summary metrics.

---

## Guided Hands-On Lab Worksheets (Day 2)

### **Lab 2.1: Multi-GPU Distributed Fine-Tuning (Single Node)**
* **Objective:** Scale fine-tuning across 4 GPUs on a single node using PyTorch Fully Sharded Data Parallel (FSDP).
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day2 ft-single --method=lora       # LoRA
    ./run.sh day2 ft-single --method=qlora      # QLoRA
    ./run.sh day2 ft-single --method=qat-lora  # QAT-LoRA
    ```
  * Direct Slurm submission:
    ```bash
    # LoRA (Container):
    cd day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_container && sbatch job_multiGPU_QA_LoRA.sh
    # LoRA (EESSI):
    cd day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_eessi && sbatch job_multiGPU_QA_LoRA_eessi.sh
    # QLoRA (4 GPUs):
    cd day2_multi_gpu/fine-tuning/ft_single_node/qlora_slurm && sbatch job_multiGPU_QA_QLoRA.sh
    # QAT-LoRA (4 GPUs):
    cd day2_multi_gpu/fine-tuning/ft_single_node/qat-lora_slurm && sbatch job_multiGPU_QA_qat-LoRA.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/fine-tuning/ft_single_node/<method>_slurm/out/*.out`
  * **Training Loss & Metrics:** `results/checkpoint_results/<RUN_NAME>/logs/`
  * **Checkpoints:** `results/checkpoint_results/<RUN_NAME>/checkpoints/`
* **Student Focus:** Use `./run.sh monitor <JobID>` to observe balanced workload distribution and memory utilization across all 4 GPUs simultaneously.

---

### **Lab 2.2: Multi-Node Distributed Scaling**
* **Objective:** Scale training across multiple cluster nodes with optimized Slingshot interconnect and AWS-OFI-NCCL network bindings.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day2 ft-multi --method=lora       # Multi-node LoRA
    ./run.sh day2 ft-multi --method=qlora      # Multi-node QLoRA
    ./run.sh day2 ft-multi --method=qat-lora  # Multi-node QAT-LoRA
    ```
  * Direct Slurm submission:
    ```bash
    # LoRA (Container):
    cd day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_container && sbatch job_multiNode_QA_LoRA.sh
    # LoRA (EESSI):
    cd day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_eessi && sbatch job_multiNode_QA_LoRA_eessi.sh
    # QLoRA (Multi-Node):
    cd day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm && sbatch job_multiNode_QA_QLoRA.sh
    # QAT-LoRA (Multi-Node):
    cd day2_multi_gpu/fine-tuning/ft_multi_node/qat-lora_slurm && sbatch job_multiNode_QA_qat-LoRA.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/fine-tuning/ft_multi_node/<method>_slurm/out/*.out`
  * **Checkpoints:** `results/checkpoint_results/<RUN_NAME>/checkpoints/`

---

### **Lab 2.3: Distributed Profiling & Communication Bottleneck Analysis**
* **Objective:** Profile multi-GPU training to identify inter-GPU collective communication overhead in FSDP.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day2 profile --method=lora
    ./run.sh day2 profile --method=qlora
    ./run.sh day2 profile --method=qat-lora
    ```
  * Direct Slurm submission:
    ```bash
    cd day2_multi_gpu/profiling
    sbatch job_multiGPU_QA_LoRA.sh       # Profile LoRA
    sbatch job_multiGPU_QA_QLoRA.sh      # Profile QLoRA
    sbatch job_multiGPU_QA_qat-LoRA.sh   # Profile QAT-LoRA
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/profiling/out/*.out`
  * **Per-Rank Traces:** Outputs generated per GPU rank under `results/profiling_results/<RUN_NAME>/profiles/` (`rank0_...`, `rank1_...`, etc.)
* **Viewing Distributed Traces Locally:**
  ```bash
  scp -r USERNAME@olivia.sigma2.no:/cluster/projects/nn9997k/USERNAME/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_multi_device/profiles .
  open profiles/rank0_memory-timeline.html
  ```
* **Key Analytical Questions for Students:**
  * What proportion of iteration time is spent in NCCL collective calls (`all_reduce`, `reduce_scatter`) vs. local matrix multiplication compute?
  * How does NVLink intra-node communication efficiency compare against single-GPU kernel execution?

---

### **Lab 2.4: Multi-GPU Distributed Evaluation**
* **Objective:** Evaluate multi-GPU fine-tuned checkpoints on the MMLU benchmark across 4 GPUs.
* **Checkpoint Selection:** By default, evaluation scripts evaluate the Day 1 LoRA checkpoint (`results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0`). To evaluate your Day 2 multi-device checkpoint, pass `LORA_PATH="${PROJECT_DIR}/results/checkpoint_results/llama3.2-1b_lora_multi_device/checkpoints/epoch_0"`.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day2 eval         # Evaluate LoRA adapter (merges adapter first)
    ./run.sh day2 eval --base  # Evaluate base model baseline
    ```
  * Direct Slurm submission:
    ```bash
    cd day2_multi_gpu/evaluation && sbatch job_multiGPU_LoRA.sh
    # or base model:
    cd day2_multi_gpu/evaluation && sbatch job_multiGPU_No_LoRA.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/evaluation/out/*.out`
  * **Benchmark Metrics:** `results/evaluation_results/llama3.2-1b-mmlu_multigpu/`

---

### **Lab 2.5: Post-Training Quantization (PTQ)**
* **Objective:** Quantize base model weights using diverse quantization backends without retraining.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day2 quantize bnb4      # BitsAndBytes 4-bit
    ./run.sh day2 quantize bnb8      # BitsAndBytes 8-bit
    ./run.sh day2 quantize torchao   # TorchAO
    ./run.sh day2 quantize gptq      # GPTQModel
    ```
  * Direct Slurm submission:
    ```bash
    cd day2_multi_gpu/quantization/quat_bitsandbytes && sbatch job_bnb_4bit.sh
    cd day2_multi_gpu/quantization/quat_torchao && sbatch job_torchao.sh
    cd day2_multi_gpu/quantization/quat_gptqmodel && sbatch job_gptq.sh
    ```
* **Where to Inspect Outputs:**
  * **Slurm Job Logs:** `day2_multi_gpu/quantization/<backend>/out/*.out`
  * **Saved Quantized Model Directory:** `${SHARED_DIR}/models/Llama-3.2-1B-Instruct-<backend>`
* **Student Focus:** Inspect file sizes of the generated model weights and compare the on-disk footprint across FP16, INT8, and INT4.

---

### **Lab 2.6: High-Throughput Serving & Interactive Terminal Chat with vLLM**
* **Objective:** Deploy an OpenAI-compatible vLLM serving engine with dynamic LoRA adapter loading and connect using an interactive terminal chat interface.
* **Adapter Selection:** Serving scripts default to loading the Day 1 LoRA checkpoint (`results/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0`). To serve your Day 2 multi-device checkpoint instead, pass `LORA_PATH="${PROJECT_DIR}/results/checkpoint_results/llama3.2-1b_lora_multi_device/checkpoints/epoch_0" ./run.sh day2 serve`.
* **Port Customization:** In a classroom environment with multiple users, specify a distinct port to avoid collisions: `PORT=8001 ./run.sh day2 serve`.
* **Launch Commands:**
  1. Start server:
     ```bash
     ./run.sh day2 serve
     # or direct:
     cd day2_multi_gpu/serving/multi_device/vllm_container && sbatch serve_vllm.sh
     ```
  2. Monitor startup:
     ```bash
     ./run.sh monitor <JobID>
     # or inspect:
     cat day2_multi_gpu/serving/multi_device/vllm_container/out/*.out
     ```
  3. Connect chat client:
     ```bash
     ./run.sh day2 chat
     # or direct:
     cd day2_multi_gpu/serving/multi_device/vllm_container && ./chat.sh
     ```
* **Student Focus:** Interact with your fine-tuned QA model directly from the command line and observe how vLLM handles token streaming.

---

### **Lab 2.7: Distributed High-Throughput Batch Inference**
* **Objective:** Benchmark batch inference throughput using vLLM tensor parallelism across multiple GPUs.
* **Launch Commands:**
  * Fast launcher (from repo root):
    ```bash
    ./run.sh day2 inference               # Container
    ./run.sh day2 inference --env=eessi   # EESSI
    ```
  * Direct Slurm submission:
    ```bash
    cd day2_multi_gpu/inference && sbatch slurm_vllm_distributed_container.sh
    # or with EESSI:
    cd day2_multi_gpu/inference && sbatch slurm_vllm_distributed_eessi.sh
    ```
* **Where to Inspect Outputs:**
  * **Throughput & Latency Metrics:** `day2_multi_gpu/inference/out/*.out`
* **Student Focus:** Inspect the generated tokens/second throughput and per-token generation latency reported in the Slurm log.
