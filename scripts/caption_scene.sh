export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/model_cjh/hf_cache

DATA_ROOT=/home/user1/data/aigc/SynCamVideo-Dataset
MODEL_PATH=/data/model_cjh/Qwen/Qwen3.5-9B

# test
python src/caption/caption_scene_level_qwen35.py \
  --model_path /data/model_cjh/Qwen/Qwen3.5-9B \
  --preview_dir data/scene_previews/train \
  --out data/captions/captions_scene_level_train_test.json \
  --max_words 40 \
  --limit 5

python src/caption/caption_scene_level_qwen35.py \
  --model_path /data/model_cjh/Qwen/Qwen3.5-9B \
  --preview_dir data/scene_previews/train \
  --out data/captions/captions_scene_level_train.json \
  --max_words 40 \
  --resume

python src/caption/caption_scene_level_qwen35.py \
  --model_path /data/model_cjh/Qwen/Qwen3.5-9B \
  --preview_dir data/scene_previews/val \
  --out data/captions/captions_scene_level_val.json \
  --max_words 40 \
  --resume

python scripts/03_check_captions.py \
  --caption_json data/captions/captions_scene_level_train.json \
  --max_words 40

python scripts/03_check_captions.py \
  --caption_json data/captions/captions_scene_level_val.json \
  --max_words 40