# Building a Custom Singularity Container for PyTorch

This guide explains how to build a custom PyTorch container using **Apptainer (formerly Singularity)** on the Olivia supercomputer.

---

## Steps

### 1. Launch an interactive session
```bash
srun -A nn9970k -p accel --nodes=1 --gpus=1 --mem-per-gpu=96G --time=00:10:00 --reservation=llm_course --pty bash -i
```

### 2. Pull the Base PyTorch Container
To accelerate pulling the container from the NVIDIA NGC catalog (and to prevent home-directory quota issues):

```bash
mkdir -p /cluster/work/projects/nn9970k/$USER/llm-hpc-course/tmp
export APPTAINER_TMPDIR=/cluster/work/projects/nn9970k/$USER/llm-hpc-course/tmp
export APPTAINER_CACHEDIR=/cluster/work/projects/nn9970k/$USER/llm-hpc-course/tmp

export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD=<Your-NVIDIA-Password>
apptainer remote login --username='$oauthtoken'
```

- [NVIDIA PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?version=26.02-py3)
- [PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html#rel-25-05)

Download the official NVIDIA PyTorch container (pinned for reproducibility):
```bash
apptainer pull --arch arm64 --disable-cache pytorch_25.05_cuda12.9_base_arm.sif \
                 docker://nvcr.io/nvidia/pytorch:25.05-py3
```
- The flag `--arch arm64` specifies the architecture for the Olivia supercomputer.
- The flag `--disable-cache` forces a clean and fresh download (see also [Sigma2 Documentation](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#downloading-containers)).

### 3. Add Extra Packages
Update the container with additional packages as defined in the definition file:
`pytorch2.8_cu2.9_py3.12_arm.def`

### 4. Build the Custom Container
Use the definition file to build a new image:
```bash
apptainer build --ignore-fakeroot-command \
    pytorch_25.05_cuda12.9_arm_custom.sif \
    pytorch2.8_cu2.9_py3.12_arm.def
```

**Output Files:**
- Base image: `pytorch_25.05_cuda12.9_base_arm.sif`
- Custom image: `pytorch_25.05_cuda12.9_arm_custom.sif`

---

## Testing the Container

### 1. Navigate to the Working Directory
```bash
cd /cluster/work/projects/nn9970k/$USER/llm-hpc-course/apptainer
```

### 2. Start an Interactive Shell
Run the container with GPU support and bind the project directories:
```bash
./launch_shell.sh
```

### 3. Run Checks Inside the Container
- **Check CUDA Availability**:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- **Show PyTorch Build Configuration**:
  ```bash
  python -c "import torch; print(torch.__config__.show())"
  ```
- **Verify NCCL Linkage**:
  ```bash
  python -c "import torch; print(torch.cuda.nccl.version())"
  ```
- **Check Installed Packages**:
  ```bash
  tune ls
  ```

---

## Fine‑tuning Setup

Once you have successfully launched the PyTorch container, you can set up fine‑tuning for **LLaMA 3.2-1B-Instruct** using the provided training recipes.

### 1. Download the Model
Specify the path to the cache:
```bash
mkdir -p /cluster/work/projects/nn9970k/$USER/llm-hpc-course/.cache/huggingface
export HF_HOME=/cluster/work/projects/nn9970k/$USER/llm-hpc-course/.cache/huggingface
```

Use `tune` to fetch the pretrained weights (if not already present):
```bash
tune download meta-llama/Llama-3.2-1B-Instruct \
  --output-dir /cluster/work/projects/nn9970k/$USER/llm-hpc-course/data/Llama-3.2-1B-Instruct \
  --ignore-patterns "original/consolidated*" \
  --hf-token <your-hugging-face-token>
```

> [!NOTE]
> Pre-trained weights are already available at: `/cluster/work/projects/nn9970k/$USER/llm-hpc-course/data/Llama-3.2-1B-Instruct`

### 2. Copy Configuration Files
Copy the built-in configuration files for single GPU and multiple GPUs:
```bash
tune cp llama3_2/1B_lora_single_device .
tune cp llama3_2/1B_lora_multi_device .
```

### 3. Copy Fine‑Tuning Recipe Scripts
```bash
tune cp lora_finetune_single_device .
tune cp lora_finetune_distributed .
```
