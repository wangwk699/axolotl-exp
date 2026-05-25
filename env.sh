# Axolotl project environment
# 使用前：cd /home/wangwenkang/axolotl-exp     
#        source .venv/bin/activate   
#        source env.sh
export CUDA_HOME=/home/wangwenkang/axolotl-exp/.cuda
export PATH=$CUDA_HOME/bin:$PATH
# vLLM / torch wheels ship libcudart in site-packages/nvidia/cuda_runtime
_VENV_NVIDIA_LIB="${VIRTUAL_ENV:+$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib}"
if [[ -n "$_VENV_NVIDIA_LIB" && -d "$_VENV_NVIDIA_LIB" ]]; then
  export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _VENV_NVIDIA_LIB
export UV_TORCH_BACKEND=cu128
export AXOLOTL_DO_NOT_TRACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
