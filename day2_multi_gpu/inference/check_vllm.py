#!/usr/bin/env python3
"""
Script to verify vLLM, PyTorch, CUDA, and NCCL installation status.
"""

import sys
import torch

def print_section(title):
    print(f"\n{'='*40}")
    print(f" {title}")
    print(f"{'='*40}")

def main():
    print_section("System & Python Info")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    # 1. Check PyTorch
    print_section("PyTorch (torch)")
    try:
        print(f"Version: {torch.__version__}")
        print(f"CUDA Build Version: {torch.version.cuda}")
        print(f"cuDNN Build Version: {torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

    # 2. Check vLLM
    print_section("vLLM")
    try:
        import vllm
        print(f"Version: {getattr(vllm, '__version__', 'Unknown/Dev Build')}")
        
        # Check if vLLM can see CUDA
        # Note: vLLM relies heavily on torch.cuda
        if hasattr(vllm, 'utils'):
            # Some versions expose cuda checks in utils
            pass 
    except ImportError:
        print("❌ vLLM is NOT installed or not in PYTHONPATH")
    except Exception as e:
        print(f"⚠️ Error importing vLLM: {e}")

    # 3. Check CUDA Availability & Devices
    print_section("CUDA Hardware")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: Yes")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n--- GPU {i} ---")
            print(f"Name: {props.name}")
            print(f"Compute Capability: {props.major}.{props.minor}")
            print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("❌ CUDA Available: No")
        print("⚠️ vLLM requires a CUDA-compatible GPU to function.")

    # 4. Check NCCL
    print_section("NCCL (NVIDIA Collective Communications Library)")
    try:
        # PyTorch bundles NCCL; this checks the bundled version
        nccl_version = torch.cuda.nccl.version()
        print(f"✅ NCCL Version: {nccl_version}")
    except AttributeError:
        print("⚠️ NCCL version not directly exposed via torch.cuda.nccl.version()")
        print("   (Usually bundled with PyTorch build)")
    except Exception as e:
        print(f"❌ Error checking NCCL: {e}")

    # 5. Quick vLLM Import Test
    print_section("vLLM Import Smoke Test")
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM core modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import vLLM core modules: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error during import: {e}")

    print("\n" + "="*40)
    print(" Check Complete")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
