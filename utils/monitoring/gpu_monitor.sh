#!/bin/bash
# Usage: ./monitor_multiGPU.sh <JOB_ID>
# To stop the script, press Ctrl+C

if [ -z "$1" ]; then
    echo -e "\e[31mError: Missing Job ID\e[0m"
    echo "Usage: $0 <JobID>"
    exit 1
fi

JOB_ID="$1"

# Check if job exists
if ! scontrol show job "$JOB_ID" > /dev/null 2>&1; then
    echo -e "\e[31mJob ID $JOB_ID not found or no longer active.\e[0m"
    exit 1
fi

# Extract Node
NODE=$(scontrol show job "$JOB_ID" | grep -oP 'NodeList=\K\S+' | grep -v '(null)' | head -n 1)

if [ -z "$NODE" ]; then
    echo -e "\e[31mFailed to resolve a valid node for Job ID $JOB_ID\e[0m"
    exit 1
fi

echo -e "\e[32mConnecting to node: $NODE (Job: $JOB_ID)\e[0m"
echo -e "\e[33mPress Ctrl+C at any time to stop monitoring and exit.\e[0m"
echo ""

# Connect to node via SSH. 
# We use an unquoted EOF so $JOB_ID and $NODE expand immediately, 
# but we escape the AWK variables (\$) so they survive the trip.
ssh -tt "$NODE" << EOF

# 1. Dynamically create the AWK formatting script on the compute node
cat << 'AWK_EOF' > $PWD/monitor_${JOB_ID}.awk
BEGIN { 
    FS=", " 
    printf "\033[1;36m%-4s | %-22s | %-8s | %-19s | %-6s | %-8s\033[0m\n", "GPU", "Model", "Util", "VRAM (Used/Total)", "Temp", "Power"
    print "\033[1;36m---------------------------------------------------------------------------------\033[0m"
}
{
    # Print the live metrics with color coding
    printf "\033[1;32m%-4s\033[0m | %-22s | %-6s | %6s / %-6s MiB | %-4s°C | %-4s W\n", \$1, \$2, \$3"%", \$4, \$5, \$6, \$7
}
AWK_EOF

# 2. Run watch directly inside the job's cgroup
# -t hides the watch header, -c enables ANSI color rendering
srun --jobid=${JOB_ID} --overlap --pty watch -n 1 -t -c "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | awk -f $PWD/monitor_${JOB_ID}.awk"

# 3. Clean up the temporary file when the user presses Ctrl+C
rm -f $PWD/monitor_${JOB_ID}.awk
exit
EOF
