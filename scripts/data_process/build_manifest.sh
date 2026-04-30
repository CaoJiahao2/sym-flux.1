# 小规模测试，避免一上来扫全量数据
# MAX_TRAIN_SCENES=2 MAX_VAL_SCENES=1 FRAME_STRIDE=8 bash scripts/01_build_manifest.sh

# 跑全量,注意要修改对应的输出路径
FRAME_STRIDE=10 MAX_ANGLE=15 NUM_VIEWS=2 bash scripts/01_build_manifest.sh

FRAME_STRIDE=10 MAX_ANGLE=30 NUM_VIEWS=2 bash scripts/01_build_manifest.sh

FRAME_STRIDE=10 MAX_ANGLE=45 NUM_VIEWS=2 bash scripts/01_build_manifest.sh

FRAME_STRIDE=10 MAX_ANGLE=15 NUM_VIEWS=4 bash scripts/01_build_manifest.sh

FRAME_STRIDE=10 MAX_ANGLE=30 NUM_VIEWS=4 bash scripts/01_build_manifest.sh

FRAME_STRIDE=10 MAX_ANGLE=45 NUM_VIEWS=4 bash scripts/01_build_manifest.sh