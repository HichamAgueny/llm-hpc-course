#!/bin/bash -e
#Sets the base project directory on the HPC cluster
export PROJECT_DIR="/cluster/work/projects/nn9997k"

#Defines your working directory within the project.
export MyWD="$PROJECT_DIR/$USER/llm-hpc-course"

#Sets the target mount point inside the container.
export CONTAINER_WD="/workspace"

#Creates a subdirectory path for storing Apptainer image
CONTAINER_DIR="${MyWD}/apptainer"

#Defines the full path to the container image file.
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.05_cuda12.9_arm_custom.sif"

#Launches an interactive shell session inside the container.
#The --nv flag enables NVIDIA GPU support
apptainer shell --nv \                
      -B "${MyWD}:${CONTAINER_WD}" \ #Mounts your host work directory into the container. 
      -B $PROJECT_DIR \              #Mounts the entire project directory (accessing shared datasets, etc)
      --env MyWD="$PROJECT_DIR/$USER/llm-hpc-course" \  #Passes the MyWD environment variable into the container.
      "${APPTAINER_SIF}"

