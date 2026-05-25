#!/usr/bin/env bash
# Clone external repos for PTQ/QAT (run once).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/external"

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ ! -d "$dir/.git" ]]; then
    git clone --depth 1 "$url" "$dir"
  else
    echo "Exists: $dir"
  fi
}

clone_if_missing https://github.com/mit-han-lab/smoothquant.git "$EXT/smoothquant"
clone_if_missing https://github.com/OpenGVLab/OmniQuant.git "$EXT/OmniQuant"
clone_if_missing https://github.com/yuhuixu1993/qa-lora.git "$EXT/qa-lora"

echo ""
echo "External repos cloned under $EXT"
echo "Create isolated conda envs from third_party_envs/*.yml when needed:"
echo "  conda env create -f third_party_envs/smoothquant.yml"
echo "  conda env create -f third_party_envs/omniquant.yml"
echo "  conda env create -f third_party_envs/qalora.yml"
