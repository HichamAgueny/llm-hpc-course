#!/bin/bash -e
#Sets the base project directory on the HPC cluster
export PROJECT_DIR="/cluster/work/projects/nn9970k"

#Defines your working directory within the project.
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"

#Sets the target mount point inside the container.
export CONTAINER_WD="/workspace"

#Creates a subdirectory path for storing Apptainer image
CONTAINER_DIR="${MyWD}/apptainer"

#Defines the full path to the container image file.
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

#Launches an interactive shell session inside the container.
#The --nv flag enables NVIDIA GPU support
#Mounts your host work directory into the container.
#Mounts the entire project directory (accessing shared datasets, etc)
#Passes the MyWD environment variable into the container.
apptainer shell --nv \                
      -B "${MyWD}:${CONTAINER_WD}" \  
      -B $PROJECT_DIR \              
      --env MyWD="$PROJECT_DIR/$USER/llm-hpc-course" \  
      "${APPTAINER_SIF}"

