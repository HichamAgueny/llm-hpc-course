BEGIN { 
    FS=", " 
    printf "\033[1;36m%-4s | %-22s | %-8s | %-19s | %-6s | %-8s\033[0m\n", "GPU", "Model", "Util", "VRAM (Used/Total)", "Temp", "Power"
    print "\033[1;36m---------------------------------------------------------------------------------\033[0m"
}
{
    # Print the live metrics with color coding
    printf "\033[1;32m%-4s\033[0m | %-22s | %-6s | %6s / %-6s MiB | %-4s°C | %-4s W\n", $1, $2, $3"%", $4, $5, $6, $7
}
