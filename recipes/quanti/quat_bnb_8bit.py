# adapted from https://huggingface.co/docs/transformers/en/quantization/bitsandbytes

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Quantize Hugging Face models to 8-bit precision and save locally.")
    
    # Required path arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="The Hugging Face model ID or local directory path to the base model.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="The local directory where the 8-bit quantized model will be saved.")
    
    # Optional hardware and precision configurations
    parser.add_argument("--device_map", type=str, default="auto", 
                        help="Device mapping strategy (e.g., 'auto', 'cuda:0').")
    parser.add_argument("--dtype", type=str, default="auto", 
                        help="Data type for non-quantized layers ('auto', 'float16', 'bfloat16').")

    args = parser.parse_args()

    # 1. Initialize 8-bit quantization config
    print("--> Initializing 8-bit quantization configuration...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Handle torch_dtype conversions for unquantized modules (e.g., LayerNorm)
    torch_dtype = args.dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    # 2. Load the Tokenizer
    print(f"--> Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 3. Load the Model in 8-bit
    print(f"--> Loading model in 8-bit from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype
    )

    # 4. Check memory footprint
    footprint_bytes = model.get_memory_footprint()
    footprint_gb = footprint_bytes / (1024 ** 3)
    print(f"--> Model successfully loaded!")
    print(f"--> Memory Footprint (8-bit): {footprint_gb:.2f} GB ({footprint_bytes} bytes)")

    # 5. Save the 4-bit model and tokenizer locally
    print(f"--> Saving 4-bit model locally to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("--> Local export complete!")

if __name__ == "__main__":
    main()
