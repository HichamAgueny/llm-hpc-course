# Parameter‑Efficient Fine‑Tuning (PEFT) Configurations

This directory contains YAML configuration files for fine‑tuning LLMs using **LoRA** (Low‑Rank Adaptation) and **QLoRA** (Quantized LoRA).

## Directory Structure

| Folder | Description |
| :--- | :--- |
| `lora/` | High‑rank adaptation configurations for full precision or BF16 training. |
| `qlora/` | Quantized 4‑bit configurations for memory‑efficient training. |

---

## Configuration Files

The configs are categorized by model type and device strategy:

### Single Device (Single GPU)
- `llama3_2_1B_lora_single_device_XSum.yaml`: LoRA fine‑tuning on the XSum dataset.
- `llama3_2_1B_qlora_single_device_profiling.yaml`: QLoRA setup for performance profiling.

### Multi Device (Distributed)
- `llama3_1_8B_lora_multi_device_alpaca_gpt4.yaml`: Distributed LoRA on Alpaca GPT-4 dataset.
- `llama3_1_8B_qlora_multi_device_APIdataset.yaml`: Distributed QLoRA on a custom API dataset.

---

## Usage

These configuration files are typically passed to the training scripts located in the `recipes/` folder via the `--config` flag.

Example:
```bash
python recipes/single_device/lora_finetune_single_device.py \
    --config configs/lora/llama3_2_1B_lora_single_device_XSum.yaml
```

> [!TIP]
> To override the checkpointer directory while launching training, you can run:
> ```bash
> python recipes/single_device/lora_finetune_single_device.py \
>     --config configs/lora/llama3_2_1B_lora_single_device_XSum.yaml \
>     checkpointer.checkpoint_dir=<YOUR_CHECKPOINT_DIR>
> ```
