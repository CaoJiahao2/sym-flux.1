python src/caption/make_scene_previews.py \
  --dataset_root /home/user1/data/aigc/flux/data/SynCamVideo-Dataset \
  --split train \
  --out_dir data/scene_previews/train \
  --thumb_size 256

  python src/caption/make_scene_previews.py \
  --dataset_root /home/user1/data/aigc/flux/data/SynCamVideo-Dataset \
  --split val \
  --out_dir data/scene_previews/val \
  --thumb_size 256