#!/usr/bin/env bash
# Re-login to HuggingFace (方案 B). Uses hf-mirror when HF_ENDPOINT is set in .bashrc.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"

if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN="$HF_TOKEN"
elif [[ -n "${1:-}" ]]; then
  TOKEN="$1"
else
  echo "Usage: HF_TOKEN=hf_xxx bash scripts/setup_hf_auth.sh"
  echo "   or: bash scripts/setup_hf_auth.sh hf_xxx"
  exit 1
fi

if [[ -f "$TOKEN_FILE" ]]; then
  cp "$TOKEN_FILE" "${TOKEN_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "Backed up old token to ${TOKEN_FILE}.bak.*"
fi

# Prefer mirror (same as ~/.bashrc); fall back to official hub.
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "Logging in via $ENDPOINT ..."
HF_ENDPOINT="$ENDPOINT" hf auth login --token "$TOKEN" --force

echo "Verifying ..."
HF_ENDPOINT="$ENDPOINT" hf auth whoami
echo "HuggingFace login OK."
