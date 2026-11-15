# Building a Custom Singularity/Apptainer Container for PyTorch

This guide explains how to build a custom PyTorch container using **Apptainer (formerly Singularity)** on the Olivia supercomputer.

> **Participants:** you do not need to build anything — the course containers are copied into your
> course folder by `./setup_env.sh`. This guide shows how they were made.

---

## Steps

### 1. Launch an interactive session
```bash
srun -A nn9997k --reservation=software -p accel --nodes=1 --gpus=1 --mem-per-cpu=8G --time=00:30:00 --pty bash -i
```

### 2. Pull the Base PyTorch Container
To accelerate pulling the container from the NVIDIA NGC catalog (and to prevent home-directory quota issues):

```bash
mkdir -p /cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/tmp
export APPTAINER_TMPDIR=/cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/tmp
export APPTAINER_CACHEDIR=/cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/tmp

export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD=<Your-NVIDIA-Password>
apptainer remote login --username='$oauthtoken'
```

- [NVIDIA PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?version=26.02-py3)
- [PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html#rel-25-05)

Download the official NVIDIA PyTorch container (pinned for reproducibility):
```bash
apptainer pull --arch arm64 pytorch_25.08_cuda13.0_arm.sif \
                 docker://nvcr.io/nvidia/pytorch:25.08-py3
```
- The flag `--arch arm64` specifies the architecture for the Olivia supercomputer.
- If disabling cache is desired then add the flag `--disable-cache` to force a clean download (see also [Sigma2 Documentation](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#downloading-containers)).
- After the image has been pulled successfully, temporary cache files can be removed:
```bash
rm -rf /cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/tmp/*
```

### 3. Build the Custom Container
Use the definition file to build a new image with required PEFT and torchtune packages:
```bash
apptainer build --ignore-fakeroot-command \
    pytorch_25.08_cuda13.0_arm_custom.sif \
    pytorch2.10_cuda13.0.2_py3.12_arm.def
```

**Output Files:**
- Base image: `pytorch_25.08_cuda13.0_arm.sif`
- Custom image: `pytorch_25.08_cuda13.0_arm_custom.sif`

---

## Testing the Container

### 1. Navigate to the Working Directory
```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer
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

## Shared Models & Datasets

Pre-downloaded base models and datasets for the course are available under the shared project directory:
```text
/cluster/work/projects/nn9971k/$USER/llm-hpc-course/shared/models/Llama-3.2-1B-Instruct
/cluster/work/projects/nn9971k/$USER/llm-hpc-course/shared/datasets/GPT-4-LLM/alpaca_gpt4_data.json
```

HuggingFace caching should be pointed to your personal share directory:
```bash
mkdir -p /cluster/work/projects/nn9971k/$USER/llm-hpc-course/hf_cache
export HF_HOME=/cluster/work/projects/nn9971k/$USER/llm-hpc-course/hf_cache
```
