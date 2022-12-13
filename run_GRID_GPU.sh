#!/bin/bash
#

cd /home/ewang96/CS1430/Editing_CVFinal/CVFinal

source /home/ewang96/miniconda3/bin/activate tf_gpu1

FILE=$1
# Print out the python file so we have a record. 
# Useful when iterating a file so you can keep track of what 
# You are doing
# Shift increments the argv values.
shift
# Tensorpack specific setting to cache ZMQ pipes in /ltmp/ 
# for speed boost
export TENSORPACK_PIPEDIR=/ltmp/
export PATH="/home/ewang96/.local/bin:${PATH}"
# If you need to compile any CUDA kernels do it on the local FS 
# so it happens faster
export CUDA_CACHE_PATH=/ltmp/

export LD_LIBRARY_PATH="/home/ewang96/miniconda3/lib:${LD_LIBRARY_PATH}"
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
# Runs the python file passing all args and pipes n into the file. 
# -u tells python to not buffer the output to so it is printed
# more often.

which python
pwd
echo -e "n\n" | python -u $FILE $@

#Reminder
# qhost to look at machines and the current usage
# qstat to view your running jobs
# qstat -u '*' to view everyone's running jobs
# qsub to run the job
