# Distributed Profiling (Multi‑GPU)

This folder contains instructions for profiling GPU performance and communication overhead in a multi‑GPU (distributed) environment. Scaling LLMs efficiently requires understanding how GPUs synchronize and share memory.

---

## 1. Enabling Distributed Profiling

To profile a distributed job, enable the profiler in your multi‑device YAML configuration (located in the `configs/` directory).

A minimal example (extracted from `configs/lora/llama3_1_8B_lora_multi_device_profiling.yaml`):

```yaml
# Profiler (enabled)
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: True

  # Output directory of trace artifacts
  output_dir: /cluster/work/projects/nn9997k/$USER/llm-hpc-course/data/profiling_outputs/multi_gpu

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

### Distributed Considerations:
- **Multiple Traces**: Each GPU rank will generate its own trace file (e.g., `rank0_...`, `rank1_...`).
- **Communication Overhead**: Profiling multi‑GPU runs captures collective operations like `AllGather` and `ReduceScatter` (FSDP/DDP), which are critical for debugging scaling issues.

---

## 2. Submitting a Multi‑GPU Profiling Job

Submit the distributed profiling job from this directory:

```bash
cd day2_multi_gpu/profiling
sbatch job_multiGPU_LoRA.sh
```

Ensure your job script is configured for multiple GPUs (e.g., `--nodes=1 --gpus-per-node=2`).

---

## 3. Analyzing Distributed Traces

Copy the resulting artifacts to your local machine:

```bash
scp -r USERNAME@olivia.sigma2.no:/cluster/work/projects/nn9997k/USERNAME/llm-hpc-course/data/profiling_outputs/multi_gpu .
```

### Viewing Results
Open the memory timelines or traces in a browser:
- `rank0_memory-timeline.html`
- `rank1_memory-timeline.html`

### TensorBoard (Recommended for Distributed)
TensorBoard's PyTorch Profiler plugin is particularly useful for comparing performance across ranks.

1. **Launch TensorBoard locally**:
   ```bash
   tensorboard --logdir=multi_gpu/
   ```
2. **Analysis**: Look at the **"Distributed"** tab within the PyTorch Profiler view to see load balancing and communication vs. computation overlap.

---

## Best Practices
- **Rank Filtering**: In TensorBoard, you can filter by rank to isolate issues on specific GPUs.
- **Scaling Analysis**: Compare traces from 2-GPU vs. 4-GPU runs to calculate your scaling efficiency.
- **Trace Size**: Distributed traces grow quickly. Limit your runs to ~10 steps.
