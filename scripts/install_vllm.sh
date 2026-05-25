#!/usr/bin/env bash
# Install vLLM cu129 wheel (matches torch cu128). PyPI default pulls cu130 → libcudart.so.13 error.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

WHEEL="/tmp/wheels/vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_x86_64.whl"
MIRROR_URL="https://ghfast.top/https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl"

mkdir -p /tmp/wheels

if [[ ! -f "$WHEEL" ]] || ! python3 -c "d=open('$WHEEL','rb').read(4); exit(0 if d==b'PK\\x03\\x04' else 1)" 2>/dev/null; then
  echo "Downloading vLLM cu129 wheel (~437MB)..."
  # Optional proxy (SSH reverse tunnel): export http_proxy/https_proxy=127.0.0.1:1080
  curl -fL --progress-bar -o "$WHEEL" "$MIRROR_URL"
  python3 -c "d=open('$WHEEL','rb').read(4); assert d==b'PK\\x03\\x04', 'bad wheel download'"
fi

echo "Installing vLLM from $WHEEL ..."
uv pip install "$WHEEL" --index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match

source env.sh
python -c "from vllm import LLM; print('vLLM', __import__('vllm').__version__, 'OK')"
