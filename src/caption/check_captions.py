import argparse
import json
import re
from pathlib import Path


BANNED = [
    "camera", "viewpoint", "view", "grid", "frame", "collage",
    "multi-view", "multiview", "left", "right", "front", "back",
    "close-up", "wide shot", "angle", "cam01", "cam05", "cam10",
]


def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json", required=True)
    parser.add_argument("--max_words", type=int, default=40)
    args = parser.parse_args()

    path = Path(args.caption_json)
    data = json.load(open(path, "r", encoding="utf-8"))

    n = len(data)
    lengths = {k: word_count(v) for k, v in data.items()}
    too_long = {k: data[k] for k, c in lengths.items() if c > args.max_words}
    banned = {
        k: v for k, v in data.items()
        if any(b in v.lower() for b in BANNED)
    }

    print(f"file: {path}")
    print(f"num captions: {n}")
    print(f"max words: {max(lengths.values()) if lengths else 0}")
    print(f"too long: {len(too_long)}")
    print(f"contains banned terms: {len(banned)}")

    print("\nExamples:")
    for k in list(data.keys())[:10]:
        print(f"{k}: {data[k]}")


if __name__ == "__main__":
    main()