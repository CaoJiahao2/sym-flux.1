GPU_IDS=1 \
MV_CKPT=outputs/flux_mvs_stage2_v4/mv_adapter_last.pt \
MANIFEST=data/val_samples.jsonl \
SAMPLE_INDEX=0 \
NUM_VIEWS=4 \
OUT=outputs/flux_mv_demo_v4.jpg \
bash scripts/30_infer_flux_mvs_manifest.sh