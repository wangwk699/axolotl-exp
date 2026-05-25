# Axolotl project environment
# 使用前：cd /home/wangwenkang/axolotl-exp     
#        source .venv/bin/activate   
#        source env.sh
export CUDA_HOME=/home/wangwenkang/axolotl-exp/.cuda
export PATH=$CUDA_HOME/bin:$PATH

# vLLM flashinfer JIT needs dev headers/libs; conda CUDA uses lib/ not lib64/
_NVIDIA_PKGS="${VIRTUAL_ENV:+$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia}"
if [[ -n "$_NVIDIA_PKGS" && -d "$_NVIDIA_PKGS" ]]; then
  _CUDA_INC="$CUDA_HOME/include"
  _CUDA_LIB64="$CUDA_HOME/lib64"
  _CUDA_STUBS="$_CUDA_LIB64/stubs"
  mkdir -p "$_CUDA_INC" "$_CUDA_STUBS"
  if [[ -e "$CUDA_HOME/lib/libcudart.so" && ! -e "$_CUDA_LIB64/libcudart.so" ]]; then
    ln -sf "../lib/libcudart.so" "$_CUDA_LIB64/libcudart.so"
  fi
  _stub="$CUDA_HOME/targets/x86_64-linux/lib/stubs/libcuda.so"
  if [[ -e "$_stub" && ! -e "$_CUDA_STUBS/libcuda.so" ]]; then
    ln -sf "../../targets/x86_64-linux/lib/stubs/libcuda.so" "$_CUDA_STUBS/libcuda.so"
  fi
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
      for _so in "$_lib"/lib*.so*; do
        [[ -e "$_so" ]] || continue
        _dst="$_CUDA_LIB64/$(basename "$_so")"
        [[ -e "$_dst" ]] || ln -sf "$_so" "$_dst"
      done
      export LD_LIBRARY_PATH="$_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  done
  _rtlib="$_NVIDIA_PKGS/cuda_runtime/lib"
  if [[ -d "$_rtlib" ]]; then
    for _so in "$_rtlib"/lib*.so*; do
      [[ -e "$_so" ]] || continue
      _dst="$_CUDA_LIB64/$(basename "$_so")"
      [[ -e "$_dst" ]] || ln -sf "$_so" "$_dst"
    done
    export LD_LIBRARY_PATH="$_rtlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  unset _CUDA_INC _CUDA_LIB64 _CUDA_STUBS _src _hdr _dst _lib _rtlib _pkg _so _stub
fi
unset _NVIDIA_PKGS

export UV_TORCH_BACKEND=cu128
export AXOLOTL_DO_NOT_TRACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
