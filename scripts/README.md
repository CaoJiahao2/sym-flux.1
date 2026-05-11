# scripts 使用说明

`scripts/` 已精简为 config-first 入口，不再维护 `train_mvs_stage1.sh`、`20_train_flux_mvs_stage1.sh` 这类阶段专用脚本。

```text
scripts/
├── 00_local_flux_env.sh     # 本地 FLUX 权重路径
├── train.sh                 # 训练入口
├── infer.sh                 # 推理入口
├── build_manifest.sh        # manifest 构建入口
├── check_forward.sh         # 模型前向检查
└── tensorboard.sh           # TensorBoard
```

---

## 1. 训练

默认配置：

```bash
bash scripts/train.sh
```

等价于：

```bash
bash scripts/train.sh configs/train/default.json
```

指定配置：

```bash
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json
```

CLI 覆盖 config：

```bash
bash scripts/train.sh configs/train/default.json --max_steps 50 --infer_every 0
```

输出目录规则：

1. config 中有非空 `output_dir`：使用 config。
2. config 中没有 `output_dir`，且 CLI 没有 `--output_dir`：脚本自动按关键参数生成目录。
3. CLI 显式传 `--output_dir`：CLI 优先。

---

## 2. 推理

默认配置：

```bash
bash scripts/infer.sh
```

指定配置：

```bash
bash scripts/infer.sh configs/val/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json
```

覆盖样本和输出：

```bash
bash scripts/infer.sh configs/val/default.json --sample_index 3 --out outputs/demo.jpg
```

---

## 3. 构建 manifest

```bash
DATASET_ROOT=/your/path/SynCamVideo-Dataset \
SPLIT=train \
OUT=data/samples/stride_8_angle_0-30_v2_train_samples.jsonl \
bash scripts/build_manifest.sh -- \
  --frame_stride 8 \
  --num_views 2 \
  --min_angle 0 \
  --max_angle 30 \
  --sampling random
```

`--sampling random` 是推荐设置，避免总是选最小角度 view 组合。

---

## 4. 检查前向

```bash
bash scripts/check_forward.sh
```

用于快速确认 FLUX + MVS wrapper 能完成一次前向。

---

## 5. TensorBoard

```bash
bash scripts/tensorboard.sh outputs/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000
```

也可以用 config 推断输出目录：

```bash
CONFIG=configs/train/default.json bash scripts/tensorboard.sh
```
