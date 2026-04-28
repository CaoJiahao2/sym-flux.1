#!/usr/bin/env bash
set -euo pipefail

# Run from flux_syncam project root.
# This does not modify data/SynCamVideo-Dataset.

FLUX_DIR=${FLUX_DIR:-third_party/flux}

mkdir -p third_party
if [ ! -d "$FLUX_DIR/.git" ]; then
  git clone https://github.com/black-forest-labs/flux "$FLUX_DIR"
else
  echo "Found existing $FLUX_DIR"
fi

python -m pip install -U pip
python -m pip install -e "$FLUX_DIR[all]"
python -m pip install decord torchvision tqdm safetensors huggingface_hub einops

# Optional local checkpoint path. If unset, use --hf_download in later scripts.
# export FLUX_DEV=/absolute/path/to/flux1-dev.safetensors
# export AE=/absolute/path/to/ae.safetensors
