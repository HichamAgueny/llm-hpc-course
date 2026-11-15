"""
python merge_and_save.py \
  --base_model_path "/workspace/shared/models/Llama-3.2-1B-Instruct" \
  --lora_adapter_path "/workspace/results/checkpoints_out/llama3_2_1B_lora_multi_device/epoch_3" \
  --output_path "/workspace/results/checkpoints_out/Llama-3.2-1B-Instruct_merged" \
  --max_shard_size "5GB"
"""

import time
import argparse 
import os
import shutil
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. Path Definitions ---
# Modified to accept an output directory and maximum file shard sizes
parser = argparse.ArgumentParser(description="Permanently merge and save a base model and LoRA adapter.")
parser.add_argument(
    "--base_model_path",
    type=str,
    required=True,
    help="Path to the directory containing your base Large Language Model (e.g., /workspace/shared/models/Llama-3.2-1B-Instruct)."
)
parser.add_argument(
    "--lora_adapter_path",
    type=str,
    required=True,
    help="Path to the directory containing your LoRA adapter weights (e.g., .../llama3_2_1B_lora_multi_device/epoch_3)."
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to the directory where the permanently merged standalone model will be saved."
)
parser.add_argument(
    "--max_shard_size",
    type=str,
    default="5GB",
    help="Maximum size limit per weight file shard (e.g., '5GB', '10GB', '20GB'). Defaults to '5GB'."
)

args = parser.parse_args()

base_model_path = args.base_model_path
lora_adapter_path = args.lora_adapter_path
output_path = args.output_path
max_shard_size = args.max_shard_size

# --- 2. Load Tokenizer ---
print(f"Loading tokenizer from: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

# --- 3. Load Base Model ---
# Changed device_map to "auto" to safely balance extreme scales (like 70B models)
# between system RAM, disk space, and GPUs without triggering Out-Of-Memory errors.
print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    low_cpu_mem_usage=True
)

# --- 4. Load LoRA Adapter on Top of the Base Model ---
print(f"Loading LoRA adapter from: {lora_adapter_path}")
model = PeftModel.from_pretrained(model, lora_adapter_path)

# --- 5. Measure and Print Time to Merge LoRA Weights ---
print("Starting LoRA weight merge operation...")
start_time = time.time()
model = model.merge_and_unload() 
end_time = time.time()
merge_duration = end_time - start_time
print(f"LoRA weights merged to base model in {merge_duration:.4f} seconds.")

# --- 6. Save the Merged Standalone Model to Disk ---
print(f"Saving fully combined weights to disk at: {output_path}...")
os.makedirs(output_path, exist_ok=True)

# Using safe_serialization writes modern .safetensors files instead of legacy pickle files.
model.save_pretrained(
    output_path, 
    max_shard_size=max_shard_size, 
    safe_serialization=True
)
print("Model weight files successfully written.")

# --- 7. Copy Required Tokenizer Assets for Torchtune Pipeline ---
print("Syncing auxiliary files and tokenizer directory structures...")

# Torchtune specifically looks for an 'original/' directory containing the raw tokenizer
original_src = os.path.join(base_model_path, "original")
original_dst = os.path.join(output_path, "original")
if os.path.exists(original_src):
    shutil.copytree(original_src, original_dst, dirs_exist_ok=True)

# Move utility JSON configurations and prompt jinja layouts from adapter path
config_files = ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
for filename in config_files:
    src_file = os.path.join(lora_adapter_path, filename)
    if os.path.exists(src_file):
        shutil.copy(src_file, os.path.join(output_path, filename))

print("\n🎉 Success! The model is fully merged, self-contained, and ready.")
print("==========================================================")

# --- 8. Auto-Generate Torchtune YAML Snippet ---
# Automatically reads the directory to eliminate manual typing typos
safetensor_files = glob.glob(os.path.join(output_path, "*.safetensors"))
filenames = sorted([os.path.basename(f) for f in safetensor_files])
