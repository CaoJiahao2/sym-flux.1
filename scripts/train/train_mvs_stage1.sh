# 基础参数定义
GPU_IDS=2
NUM_VIEWS=2
MAX_STEPS=1000
MV_ADAPTER_DIM=1024
LR=1e-4
BATCH_SIZE=1
GRAD_ACCUM=8

# 动态生成 OUTPUT_DIR
# 命名规则：outputs/flux_mvs_S{MAX_STEPS}_V{NUM_VIEWS}_D{MV_ADAPTER_DIM}
OUTPUT_DIR=outputs/flux_mvs_steps${MAX_STEPS}_views${NUM_VIEWS}_dim${MV_ADAPTER_DIM}

# 执行训练
GPU_IDS=$GPU_IDS \
TRAIN_MANIFEST=data/stride_10_angle_15_train_samples.jsonl \
NUM_VIEWS=$NUM_VIEWS \
RESOLUTION=512 \
BATCH_SIZE=$BATCH_SIZE \
GRAD_ACCUM=$GRAD_ACCUM \
MAX_STEPS=$MAX_STEPS \
LR=$LR \
MV_ADAPTER_DIM=$MV_ADAPTER_DIM \
OUTPUT_DIR=$OUTPUT_DIR \
bash scripts/20_train_flux_mvs_stage1.sh

# # 自动引用训练阶段的输出路径
# MV_CKPT=${OUTPUT_DIR}/mv_adapter_last.pt
# # 定义推理输出图片名，并放置在训练目录下
# OUT=${OUTPUT_DIR}/sample_S${MAX_STEPS}_V${NUM_VIEWS}_D${MV_ADAPTER_DIM}.jpg

# # 执行推理
# GPU_IDS=$GPU_IDS \
# MV_CKPT=$MV_CKPT \
# MANIFEST=data/stride_10_angle_15_val_samples.jsonl \
# SAMPLE_INDEX=0 \
# NUM_VIEWS=$NUM_VIEWS \
# OUT=$OUT \
# bash scripts/30_infer_flux_mvs_manifest.sh