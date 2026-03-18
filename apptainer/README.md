# Apptainer Container Setup

This directory contains the necessary files to build and run the course environment using **Apptainer**.

## Directory Contents

| File | Description |
| :--- | :--- |
| `pytorch2.8_cu2.9_py3.12_arm.def` | The Apptainer definition file used to build the container image. |
| `launch_shell.sh` | A utility script to launch an interactive shell inside the container. |
| `srun_interactive` | A Slurm script to request an interactive GPU session. |

---

## 1. Requesting a GPU Session

Before launching the container, you must request an interactive GPU session from Slurm. Use the provided `srun_interactive` script:

```bash
bash srun_interactive
```
This script requests:
- 1 GPU
- 1 Node
- 20GB Memory per CPU
- 15 minutes of wall-time

## 2. Launching the Container

Once you have a GPU allocation, you can launch the container shell:

```bash
bash launch_shell.sh
```

The script handles:
- Mounting relevant directories (`$PROJECT_DIR` and `$MyWD`).
- Setting up environment variables.
- Enabling NVIDIA GPU support (`--nv`).

---

## Building the Image (Optional)

If you need to rebuild the image from the definition file:

```bash
apptainer build pytorch_custom.sif pytorch2.8_cu2.9_py3.12_arm.def
```

> [!NOTE]
> Ensure you have sufficient quota and permissions to build images on the cluster.
