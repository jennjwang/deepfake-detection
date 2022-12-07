#!/bin/bash
#
#  Execute from the current working directory
# $ -cwd
#
# This requests a single GPU. Unless you write
# Your code in a very specific way, it will not
# benefit from additional GPUs.
# $ -l gpus=1

# This loads the virtual environment
# source /course/cs1430/tf_gpu/bin/activate
# original line above doesn't work bc there's no tf_gpu



# cd /home/ewang96/miniconda3/envs/tf_gpu

# cd /home/ewang96/CS1430/DeepFake-Detect
# cd /home/ewang96/miniconda3/envs/tf_gpu
# pwd
cd /home/ewang96/CS1430/Editing_CVFinal/CVFinal

source /home/ewang96/miniconda3/bin/activate tf_gpu1
# pip install -r requirements.txt --user

# eval "$(conda shell.bash hook)"
# source activate tf_gpu
FILE=$1
# Print out the python file so we have a record. 
# Useful when iterating a file so you can keep track of what 
# You are doing
# uncomment this next line to print the python file to stdout
# cat $FILE
# Shift increments the argv values.
shift
# Tensorpack specific setting to cache ZMQ pipes in /ltmp/ 
# for speed boost
# export PYTHONPATH="${PYTHONPATH}:/home/ewang96/miniconda3/envs/tf_gpu/lib/python3.9"
# export LD_LIBRARY_PATH="$/home/ewang96/miniconda3/envs/tf_gpu1/lib/python3.8/site-packages/tensorrt:${LD_LIBRARY_PATH}"
export TENSORPACK_PIPEDIR=/ltmp/
export PATH="/home/ewang96/.local/bin:${PATH}"
# If you need to compile any CUDA kernels do it on the local FS 
# so it happens faster
export CUDA_CACHE_PATH=/ltmp/

# export LD_LIBRARY_PATH="/home/ewang96/.local/lib/python3.9/site-packages/tensorrt/${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/ewang96/miniconda3/lib:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
# export TF_XLA_FLAGS=--tf_xla_auto_jit=2
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
# Runs the python file passing all args and pipes n into the file. 
# -u tells python to not buffer the output to so it is printed
# more often.

# conda list 
which python
pwd
echo -e "n\n" | python -u $FILE $@

#Reminder
# qhost to look at machines and the current usage
# qstat to view your running jobs
# qstat -u '*' to view everyone's running jobs
# qsub to run the job
