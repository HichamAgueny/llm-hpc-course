export APPTAINER_TMPDIR=/cluster/work/projects/nn9997k/$USER/tmp
export APPTAINER_CACHEDIR=/cluster/work/projects/nn9997k/$USER/cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

apptainer build --ignore-fakeroot-command \
    Xpytorch_25.08_cuda13.0_arm_custom.sif \
    pytorch2.10_cuda13.0.2_py3.12_arm.def

#with nohup
nohup apptainer build --ignore-fakeroot-command \
     Xpytorch_25.08_cuda13.0_arm_custom.sif \
     pytorch2.10_cuda13.0.2_py3.12_arm.def \
     > build.log 2>&1 &

