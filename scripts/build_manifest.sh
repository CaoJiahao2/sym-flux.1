# 小规模测试，避免一上来扫全量数据
# MAX_TRAIN_SCENES=2 MAX_VAL_SCENES=1 FRAME_STRIDE=8 bash scripts/01_build_manifest.sh

# 总规模测试
MAX_TRAIN_SCENES=800 MAX_VAL_SCENES=1 FRAME_STRIDE=20 bash scripts/01_build_manifest.sh
# #跑全量
# FRAME_STRIDE=8 MAX_ANGLE=60 bash scripts/01_build_manifest.sh