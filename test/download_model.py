import os
from pathlib import Path
from modelscope import snapshot_download


# MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_ID = 'Qwen/Qwen3.5-9B'
SAVE_ROOT = Path("/data/model_cjh")
# LOCAL_DIR = SAVE_ROOT / "FLUX.2-klein-base-4B"
LOCAL_DIR = SAVE_ROOT / "Qwen/Qwen3.5-9B"


def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 可选：把 ModelScope 缓存也放到 /data/model_cjh，避免占用 ~/.cache
    os.environ.setdefault("MODELSCOPE_CACHE", str(SAVE_ROOT / ".modelscope_cache"))

    print(f"Downloading ModelScope model: {MODEL_ID}")
    print(f"Target directory: {LOCAL_DIR}")

    model_dir = snapshot_download(
        model_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        revision="master",
    )

    print("\nDownload finished.")
    print(f"Returned model dir: {model_dir}")
    print(f"Local model path: {LOCAL_DIR}")


if __name__ == "__main__":
    main()