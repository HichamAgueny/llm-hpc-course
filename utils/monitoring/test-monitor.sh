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

# Connect via SSH
ssh -tt "$NODE" << EOF

# 1. Dynamically create the AWK formatting script
cat << 'AWK_EOF' > $PWD/monitor_${JOB_ID}.awk
BEGIN { 
    FS=", " 
    printf "\033[1;36m%-3s | %-18s | %-4s | %-15s | %-6s | %-8s\033[0m\n", "GPU", "Model", "Util", "VRAM (MiB)", "Temp", "Power"
    print "\033[1;36m-----------------------------------------------------------------------\033[0m"
}
{
    sub(/\r$/, "", \$7)
    sub(/^NVIDIA /, "", \$2)
    printf "\033[1;32m%-3s\033[0m | %-18s | %-3s%% | %5s / %-5s | %-4s°C | %-6s W\n", \$1, \$2, \$3, \$4, \$5, \$6, \$7
}
AWK_EOF

# 2. Custom loop instead of 'watch' to prevent terminal lockup
srun --jobid=${JOB_ID} --overlap --pty bash -c '
    # Trap ensures the cursor comes back if interrupted
    trap "tput cvvis; exit" SIGINT SIGTERM EXIT
    
    # Hide the cursor for a cleaner look
    tput civis 
    
    while true; do
        # Move cursor to top-left (smoother than full clear)
        printf "\033[H\033[J" 
        
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | awk -f $PWD/monitor_${JOB_ID}.awk
        
        sleep 1
    done
'

# 3. Cleanup remote files
rm -f $PWD/monitor_${JOB_ID}.awk
exit
EOF

# 4. Failsafe: Reset the local terminal so it never gets stuck
stty sane
tput cvvis
echo -e "\n\e[32mMonitoring ended cleanly. Your terminal is safe.\e[0m"
