# Axolotl project environment
# 使用前：cd /home/wangwenkang/axolotl-exp     
#        source .venv/bin/activate   
#        source env.sh
export CUDA_HOME=/home/wangwenkang/axolotl-exp/.cuda
export PATH=$CUDA_HOME/bin:$PATH
export UV_TORCH_BACKEND=cu128
export AXOLOTL_DO_NOT_TRACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
