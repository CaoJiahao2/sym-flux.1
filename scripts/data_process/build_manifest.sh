# Small smoke test:
# MAX_TRAIN_SCENES=2 MAX_VAL_SCENES=1 FRAME_STRIDE=8 MIN_ANGLE=0 MAX_ANGLE=30 NUM_VIEWS=2 SAMPLING=random bash scripts/01_build_manifest.sh

# Progressive manifest examples.
FRAME_STRIDE=10 MIN_ANGLE=0  MAX_ANGLE=30 NUM_VIEWS=2 SAMPLING=random bash scripts/01_build_manifest.sh
FRAME_STRIDE=10 MIN_ANGLE=15 MAX_ANGLE=45 NUM_VIEWS=2 SAMPLING=random bash scripts/01_build_manifest.sh
FRAME_STRIDE=10 MIN_ANGLE=30 MAX_ANGLE=60 NUM_VIEWS=2 SAMPLING=random bash scripts/01_build_manifest.sh

FRAME_STRIDE=10 MIN_ANGLE=0  MAX_ANGLE=30 NUM_VIEWS=4 SAMPLING=random bash scripts/01_build_manifest.sh
FRAME_STRIDE=10 MIN_ANGLE=15 MAX_ANGLE=45 NUM_VIEWS=4 SAMPLING=random bash scripts/01_build_manifest.sh
FRAME_STRIDE=10 MIN_ANGLE=30 MAX_ANGLE=60 NUM_VIEWS=4 SAMPLING=random bash scripts/01_build_manifest.sh