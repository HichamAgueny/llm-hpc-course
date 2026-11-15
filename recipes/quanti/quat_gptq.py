import argparse
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_local_calibration_data(file_path):
    """Loads lines from a local text file to use as calibration data."""
    print(f"--> Reading local calibration data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) < 100:
        print(f"⚠️ Warning: Only found {len(lines)} lines. For good GPTQ quality, aim for 500-1000 paragraphs.")
    return lines

def main():
    parser = argparse.ArgumentParser(description="Quantize models locally using GPTQModel on offline HPC clusters.")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Local directory path to the base model.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Local directory path to save the resulting model.")
    parser.add_argument("--calibration_file", type=str, required=True,
                        help="Path to a local .txt file containing calibration text.")
    
    # Precision & Hardware Parameters
    parser.add_argument("--bits", type=str, default="4", choices=["4", "8", "none", "float"],
                        help="Quantization precision.")
    parser.add_argument("--device_map", type=str, default="auto", 
                        help="Device mapping strategy.")
    parser.add_argument("--dtype", type=str, default="auto", 
                        help="Torch weights data type initialization.")
    
    # GPTQ specifics
    parser.add_argument("--group_size", type=int, default=128, help="Group size for GPTQ quantization.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for calibration processing.")

    args = parser.parse_args()

    print(f"--> Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Clean type mapping to avoid feeding strings like "auto" into GPTQ Model loader
    backend_dtype = None
    if args.dtype == "float16":
        backend_dtype = torch.float16
    elif args.dtype == "bfloat16":
        backend_dtype = torch.bfloat16

    # 1. Standard Unquantized Load Fallback
    if args.bits in ["none", "float"]:
        print(f"--> Mode: Loading model WITHOUT quantization ({args.dtype})...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map=args.device_map,
            torch_dtype=backend_dtype if backend_dtype is not None else "auto"
        )
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        
    # 2. Advanced GPTQ Quantization Load
    else:
        bit_value = int(args.bits)
        print(f"--> Mode: Initializing GPTQ {bit_value}-bit configuration...")
        quant_config = QuantizeConfig(bits=bit_value, group_size=args.group_size)

        # Load calibration data locally without internet
        calibration_dataset = load_local_calibration_data(args.calibration_file)

        print(f"--> Initializing model base with device_map='{args.device_map}'...")
        
        # Build dynamic keyword args for GPTQModel. load() using updated 'dtype' parameter name
        gptq_load_kwargs = {"device_map": args.device_map}
        if backend_dtype is not None:
            gptq_load_kwargs["dtype"] = backend_dtype
        # If args.dtype is "auto", we omit passing it so GPTQ reads the model's config native dtype safely

        model = GPTQModel.load(
            args.model_path, 
            quant_config,
            **gptq_load_kwargs
        )

        print(f"--> Executing optimization pass (Batch Size: {args.batch_size})...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)

        print(f"--> Exporting calibrated weights locally to: {args.output_path}")
        model.save(args.output_path)
        tokenizer.save_pretrained(args.output_path)

    print("--> Job execution successfully completed!")

if __name__ == "__main__":
    main()
