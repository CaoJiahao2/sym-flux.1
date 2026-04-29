export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/model_cjh/hf_cache

DATA_ROOT=/home/user1/data/aigc/SynCamVideo-Dataset
MODEL_PATH=/data/model_cjh/Qwen/Qwen3.5-9B

# run only in test
# python src/caption/caption_scene_level_qwen35.py \
#   --model_path /data/model_cjh/Qwen/Qwen3.5-9B \
#   --preview_dir data/scene_previews/train \
#   --out data/captions/captions_scene_level_train_test.json \
#   --max_words 50 \
#   --gpu_id 2 \
#   --limit 5

# train
python src/caption/caption_scene_level_qwen35.py \
  --model_path /data/model_cjh/Qwen/Qwen3.5-9B \
  --preview_dir data/scene_previews/train \
  --out data/captions/captions_scene_level_train.json \
  --max_words 50 \
  --gpu_id 2 \
  --resume

python src/caption/check_captions.py \
  --caption_json data/captions/captions_scene_level_train.json \
  --max_words 50

# val
python src/caption/caption_scene_level_qwen35.py \
  --model_path /data/model_cjh/Qwen/Qwen3.5-9B \
  --preview_dir data/scene_previews/val \
  --out data/captions/captions_scene_level_val.json \
  --max_words 50 \
  --gpu_id 4 \
  --resume

python src/caption/check_captions.py\
  --caption_json data/captions/captions_scene_level_val.json \
  --max_words 50