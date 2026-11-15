import json
import os
import sys
import glob

# Read results directory strictly from the environment variable provided by the shell script
results_dir = os.environ.get("RESULTS_DIR")

if not results_dir:
    print("ERROR: The RESULTS_DIR environment variable is not set.")
    print("Please run this script via your runner script (e.g., ./run_parse.sh)")
    sys.exit(1)

# Allow passing a specific file path as a command-line argument
if len(sys.argv) > 1:
    result_file = sys.argv[1]
else:
    pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"ERROR: No benchmark JSON files found in search path: {results_dir}")
        sys.exit(1)
    # Automatically pick the most recently modified JSON file
    result_file = max(files, key=os.path.getmtime)

print(f"-> Parsing results file: {result_file}\n")

with open(result_file, "r") as f:
    data = json.load(f)

# Extract metrics safely
duration = data.get("elapsed_time", data.get("duration", 0))
completed_reqs = data.get("completed", data.get("num_prompts", 0))
total_in_tokens = data.get("total_input_tokens", 0)
total_out_tokens = data.get("total_output_tokens", 0)

req_throughput = data.get("request_throughput", 0)
input_throughput = data.get("input_throughput", 0)
output_throughput = data.get("output_throughput", 0)

def get_stat(keys, default=0):
    curr = data
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return curr

mean_ttft = get_stat(["mean_ttft_ms"], get_stat(["ttft", "mean"], 0))
median_ttft = get_stat(["median_ttft_ms"], get_stat(["ttft", "median"], 0))
p99_ttft = get_stat(["p99_ttft_ms"], get_stat(["ttft", "p99"], 0))

mean_tpot = get_stat(["mean_tpot_ms"], get_stat(["tpot", "mean"], 0))
median_tpot = get_stat(["median_tpot_ms"], get_stat(["tpot", "median"], 0))
p99_tpot = get_stat(["p99_tpot_ms"], get_stat(["tpot", "p99"], 0))

# Print summary table
print("=" * 60)
print(f" vLLM BENCHMARK PERFORMANCE SUMMARY")
print("=" * 60)
print(f" {'Metric':<30} | {'Value':<25}")
print("-" * 60)
print(f" {'Total Successful Requests':<30} | {completed_reqs:<25}")
print(f" {'Total Duration (s)':<30} | {duration:.2f} s")
print(f" {'Total Input Tokens':<30} | {total_in_tokens}")
print(f" {'Total Output Tokens':<30} | {total_out_tokens}")
print("-" * 60)
print(f" {'Request Throughput':<30} | {req_throughput:.2f} req/s")
print(f" {'Input Token Throughput':<30} | {input_throughput:.2f} tok/s")
print(f" {'Output Token Throughput':<30} | {output_throughput:.2f} tok/s")
print("-" * 60)
print(f" {'Mean TTFT (Time to First Token)':<30} | {mean_ttft:.2f} ms")
print(f" {'Median TTFT':<30} | {median_ttft:.2f} ms")
print(f" {'P99 TTFT':<30} | {p99_ttft:.2f} ms")
print("-" * 60)
print(f" {'Mean TPOT (Time Per Output Token)':<30} | {mean_tpot:.2f} ms")
print(f" {'Median TPOT':<30} | {median_tpot:.2f} ms")
print(f" {'P99 TPOT':<30} | {p99_tpot:.2f} ms")
print("=" * 60)
