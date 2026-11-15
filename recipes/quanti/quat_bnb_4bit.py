# adapted from https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
# see QLoRA
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Quantize Hugging Face models to 4-bit and save them locally via Slurm.")
    
    # Required path arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="The Hugging Face model ID or local directory path to the base model.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="The local directory where the quantized model should be saved.")
    
    # Hardware & precision parameters
    parser.add_argument("--device_map", type=str, default="auto", 
                        help="Device mapping strategy (e.g., 'auto', 'cuda:0').")
    parser.add_argument("--dtype", type=str, default="auto", 
                        help="Data type for non-quantized layers ('auto', 'float16', 'bfloat16').")

    args = parser.parse_args()

    # Handle torch_dtype conversions for regular modules
    torch_dtype = args.dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    # 1. Dedicated 4-bit (NF4) quantization configuration
    print("--> Setting up 4-bit (NF4) quantization configuration...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if args.dtype == "auto" else torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 2. Load the Tokenizer
    print(f"--> Fetching tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 3. Load the Model with target configurations
    print(f"--> Loading model weights from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype if args.dtype != "auto" else "auto"
    )

    # 4. Check memory footprint
    footprint_bytes = model.get_memory_footprint()
    footprint_gb = footprint_bytes / (1024 ** 3)
    print(f"--> Model successfully loaded in memory!")
    print(f"--> Memory Footprint: {footprint_gb:.2f} GB ({footprint_bytes} bytes)")

    # 5. Save the Quantized Model and Tokenizer Locally
    print(f"--> Exporting quantized weights and configurations to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("--> Process successfully completed!")

if __name__ == "__main__":
    main()
