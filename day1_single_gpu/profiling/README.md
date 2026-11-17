# Profiling LLM Workloads

This folder contains tools and instructions for profiling GPU usage and memory consumption during training, as well as kernel performance. Profiling helps you analyze how efficiently your model uses system resources in an HPC environment.

---

## 1. Enabling Profiling in Configuration

To generate profiling data, you must enable and configure the profiler in your YAML configuration file (located in the `configs/` directory).

A minimal example (extracted from `configs/lora/llama3_2_1B_lora_single_device_profiling.yaml`):

```yaml
# Activations Memory
enable_activation_checkpointing: False  # True reduces memory
enable_activation_offloading: False     # True reduces memory

# Profiler (enabled)
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: True

  # Output directory for trace artifacts
  output_dir: ${profiling_dir}

  # Activities to trace
  cpu: True
  cuda: True

  # Trace options
  profile_memory: True
  with_stack: False
  record_shapes: True
  with_flops: True

  # Scheduling options
  wait_steps: 5
  warmup_steps: 3
  active_steps: 2
  num_cycles: 1
```

### Key Parameters:
- **`enabled: True`**: Turns on the PyTorch Profiler.
- **`output_dir`**: Specifies where the JSON/HTML trace files will be saved (mapped to `${PROFILING_DIR}`).
- **`wait_steps`, `warmup_steps`, `active_steps`**: Control the profiling window to capture steady-state performance.

---

## 2. Submitting a Profiling Job

Submit the profiling job script from this directory:

```bash
cd /cluster/work/projects/nn9971k/$USER/llm-hpc-course/day1_single_gpu/profiling

# Profile LoRA fine-tuning
sbatch job_singleGPU_QA_LoRA_profiling.sh

# Profile QLoRA fine-tuning
sbatch job_singleGPU_QA_QLoRA_profiling.sh
```

Profiling outputs and logs will be generated under:
```text
/cluster/work/projects/nn9971k/$USER/llm-hpc-course/results/profiling_results/<RUN_NAME>/profiles/iteration_10
/cluster/work/projects/nn9971k/$USER/llm-hpc-course/results/profiling_results/<RUN_NAME>/logs
```

---

## 3. Analyzing Results Locally

After the job completes, copy the profiling artifacts from the cluster to your local machine for analysis:

```bash
# Run this on your laptop; replace <username> with your cluster username
scp -r <username>@olivia.sigma2.no:/cluster/work/projects/nn9971k/<username>/llm-hpc-course/results/profiling_results/profiling_llama3.2-1b_lora_single_device .
```

### Viewing Traces
The profiling outputs include memory timelines and trace files (e.g., `rank0_memory-timeline.html`, `*.pt.trace.json`):

- **Memory timeline HTML**: Open directly in your browser:
  - **macOS**: `open profiles/rank0_memory-timeline.html`
  - **Linux**: `xdg-open profiles/rank0_memory-timeline.html`
  - **Windows**: `start profiles/rank0_memory-timeline.html`

### TensorBoard Integration (Optional)
For a more detailed interactive view, use the PyTorch Profiler plugin for TensorBoard:

1. **Install TensorBoard locally**:
   ```bash
   pip install torch-tb-profiler tensorboard
   ```
2. **Launch TensorBoard**:
   ```bash
   tensorboard --logdir=profiles/iteration_10
   ```
3. **Open in Browser**: Navigate to `http://localhost:6006` and select the **"PyTorch Profiler"** tab.

---

## Best Practices
- **Short Runs**: Keep the number of training steps low (e.g., 10–20 steps) to avoid generating excessively large trace files.
- **Temporary Checkpoints**: The profiling scripts point `OUTPUT_DIR` to local scratch `/tmp/${USER}_${SLURM_JOB_ID}/checkpoints` to avoid saving throwaway checkpoints to permanent storage.
- **Cleanup**: Delete old profiling runs when finished to save disk space on the cluster.
- **Multi-GPU**: Distributed runs generate one trace file per rank (e.g., `rank0_...`, `rank1_...`).
