# Profiling LLM Workloads

This folder contains tools and instructions for profiling GPU usage and memory consumption during training, as well as kernel performance. Profiling helps you analyze how efficiently your model uses system resources in an HPC environment.

---

## 1. Enabling Profiling in Configuration

To generate profiling data, you must enable and configure the profiler in your YAML configuration file (located in the `configs/` directory).

A minimal example (extracted from `configs/lora/llama3_2_1B_lora_single_device_profiling.yaml`):

```yaml
# Profiler (enabled)
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: True

  # Output directory for trace artifacts
  output_dir: /cluster/work/projects/nn9997k/$USER/llm-hpc-course/results/profiles/profiling_outputs

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
- **`output_dir`**: Specifies where the JSON/HTML trace files will be saved.
- **`wait_steps`, `warmup_steps`, `active_steps`**: Control the profiling window to capture steady-state performance.

---

## 2. Submitting a Profiling Job

Once profiling is enabled in the config, submit the job script from this directory:

```bash
cd day1_single_gpu/profiling
sbatch job_singleGPU_profiling_LoRA.sh
```

Profiling outputs will be generated in the configured `output_dir`, for example:
`/cluster/work/projects/nn9997k/$USER/llm-hpc-course/results/profiles/profiling_outputs`

---

## 3. Analyzing Results Locally

After the job completes, copy the profiling artifacts from the cluster to your local machine for analysis:

```bash
# Replace USERNAME with your actual cluster username
scp -r USERNAME@olivia.sigma2.no:/cluster/work/projects/nn9997k/USERNAME/llm-hpc-course/results/profiles/profiling_outputs .
```

### Viewing Traces
The profiling outputs are often stored as HTML files (e.g., `rank0_memory-timeline.html`). Open them in your preferred browser:

- **macOS**: `open rank0_memory-timeline.html`
- **Linux**: `xdg-open rank0_memory-timeline.html`
- **Windows**: `start rank0_memory-timeline.html`

### TensorBoard Integration (Optional)
For a more detailed interactive view, use the PyTorch Profiler plugin for TensorBoard:

1. **Install TensorBoard locally**:
   ```bash
   pip install tensorboard 
   ```
2. **Launch TensorBoard**:
   ```bash
   tensorboard --logdir=profiling_outputs/iteration_10
   ```
3. **Open in Browser**: Navigate to `http://localhost:6006` and select the **"PyTorch Profiler"** tab.

---

## Best Practices
- **Short Runs**: Keep the number of training steps low (e.g., 10-20 steps) to avoid generating excessively large trace files.
- **Cleanup**: Delete old profiling runs to save disk space on the cluster.
- **Multi-GPU**: Note that distributed runs will generate one report per rank (e.g., `rank0_...`, `rank1_...`).
