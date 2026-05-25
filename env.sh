# Axolotl project environment
# 使用前：cd /home/wangwenkang/axolotl-exp     
#        source .venv/bin/activate   
#        source env.sh
export CUDA_HOME=/home/wangwenkang/axolotl-exp/.cuda
export PATH=$CUDA_HOME/bin:$PATH

# vLLM flashinfer JIT needs dev headers (curand etc.) from pip nvidia-* wheels
_NVIDIA_PKGS="${VIRTUAL_ENV:+$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia}"
if [[ -n "$_NVIDIA_PKGS" && -d "$_NVIDIA_PKGS" ]]; then
  _CUDA_INC="$CUDA_HOME/include"
  mkdir -p "$_CUDA_INC"
  for _pkg in curand cublas cusparse cufft cusolver; do
    _src="$_NVIDIA_PKGS/$_pkg/include"
    [[ -d "$_src" ]] || continue
    for _hdr in "$_src"/*.h; do
      [[ -e "$_hdr" ]] || continue
      _dst="$_CUDA_INC/$(basename "$_hdr")"
      [[ -e "$_dst" ]] || ln -sf "$_hdr" "$_dst"
    done
    _lib="$_NVIDIA_PKGS/$_pkg/lib"
    if [[ -d "$_lib" ]]; then
      export LD_LIBRARY_PATH="$_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  done
  _rtlib="$_NVIDIA_PKGS/cuda_runtime/lib"
  if [[ -d "$_rtlib" ]]; then
    export LD_LIBRARY_PATH="$_rtlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  unset _CUDA_INC _src _hdr _dst _lib _rtlib _pkg
fi
unset _NVIDIA_PKGS

export UV_TORCH_BACKEND=cu128
export AXOLOTL_DO_NOT_TRACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
