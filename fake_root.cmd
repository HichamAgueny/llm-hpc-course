export APPTAINER_TMPDIR=/cluster/work/projects/nn9997k/$USER/tmp
export APPTAINER_CACHEDIR=/cluster/work/projects/nn9997k/$USER/cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

apptainer build --ignore-fakeroot-command \
    pytorch_25.08_cuda13.0_arm_custom.sif \
    pytorch2.10_cuda13.0.2_py3.12_arm_NEW.def
