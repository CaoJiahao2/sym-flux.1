# 小规模测试，避免一上来扫全量数据
# MAX_TRAIN_SCENES=2 MAX_VAL_SCENES=1 FRAME_STRIDE=8 bash scripts/01_build_manifest.sh

# 总规模测试
# MAX_TRAIN_SCENES=800 MAX_VAL_SCENES=1 FRAME_STRIDE=20 bash scripts/01_build_manifest.sh

# 跑全量,注意要修改对应的输出路径
OUT_BASE=data/stride_10_angle_30 FRAME_STRIDE=10 MAX_ANGLE=30 bash scripts/01_build_manifest.sh

OUT_BASE=data/stride_10_angle_15 FRAME_STRIDE=10 MAX_ANGLE=15 bash scripts/01_build_manifest.sh

OUT_BASE=data/stride_10_angle_45 FRAME_STRIDE=10 MAX_ANGLE=45 bash scripts/01_build_manifest.sh