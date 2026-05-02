# Config 配置说明

当前 `configs/` 分为两个子目录：

```text
configs/train/    # 训练配置，供 src/train_flux_multiview.py 使用
configs/val/      # 推理/验证配置，供 src/infer_flux_multiview.py 使用
```

训练和推理都支持 `--config`。命令行参数会覆盖 config 中的同名字段。

---

## 1. 训练配置

训练入口：

```bash
python src/train_flux_multiview.py --config configs/train/full_view_stage1_0_30_v4.json
```

推荐使用封装脚本：

```bash
GPU_IDS=0 bash scripts/train_with_config.sh configs/train/full_view_stage1_0_30_v4.json
```

覆盖 config 参数：

```bash
GPU_IDS=0 bash scripts/train_with_config.sh \
  configs/train/full_view_stage1_0_30_v4.json \
  --max_steps 2000 \
  --infer_every 250
```

### 训练配置列表

| 配置文件 | 用途 |
|---|---|
| `configs/train/debug_v2_100steps.json` | 2 视角、100 step 冒烟测试 |
| `configs/train/full_view_stage1_0_30_v4.json` | 4 视角，角度区间 `[0,30]`，第一阶段 |
| `configs/train/full_view_stage2_15_45_v4.json` | 4 视角，角度区间 `[15,45]`，从 stage1 继续训练 |
| `configs/train/full_view_stage3_30_60_v4.json` | 4 视角，角度区间 `[30,60]`，从 stage2 继续训练 |
| `configs/train/ablation_same_token_stage1_v4.json` | 消融实验：`mv_attn_mode=same_token` |

### 训练主要字段

#### 数据

- `train_manifest`: 训练 jsonl。
- `infer_manifest`: 训练中周期性可视化使用的 jsonl。
- `resolution`: 输入图像分辨率。
- `num_views`: 每个样本使用的视角数。
- `batch_size`: DataLoader batch size。
- `num_workers`: DataLoader worker 数。

#### 模型与 MVS

- `model_name`: FLUX 模型名，默认 `flux-dev`。
- `mv_arch`: `adapter` 或 `full_hidden`。
- `mv_adapter_dim`: adapter 隐层维度，仅 `mv_arch=adapter` 时有效。
- `mv_attn_mode`: `full_view` 或 `same_token`。
- `inject_single_blocks`: 是否在 single stream blocks 中注入 MVS。
- `single_block_stride`: single stream blocks 的注入间隔。
- `no_mv_timestep_modulation`: 是否关闭 timestep modulation。
- `mv_dropout`: MVS dropout。
- `resume_mv_ckpt`: 继续训练的 MVS adapter checkpoint。

#### 优化

- `max_steps`: optimizer steps 数量，不是 micro steps。
- `learning_rate`: AdamW learning rate。
- `weight_decay`: AdamW weight decay。
- `grad_accum`: 梯度累积步数。
- `mixed_precision`: `bf16`、`fp16` 或 `fp32`。
- `gradient_checkpointing`: 是否开启 gradient checkpointing。
- `seed`: 随机种子。

#### 正则化

- `pseudo_general_prob`: 以一定概率复制单视角为多视角，并把相机设为 identity，用于保护 FLUX 原始视觉质量。
- `pseudo_general_random_view`: pseudo general regularization 是否随机选复制源视角。
- `noise_share_ratio`: 已废弃，训练/推理均使用独立 Gaussian noise。

#### checkpoint、日志和训练中可视化

- `save_every`: 每多少 optimizer steps 保存一次 adapter。
- `log_every`: 每多少 optimizer steps 打印一次日志。
- `infer_every`: 每多少 optimizer steps 进行一次多视角推理并保存可视化图。默认 `500`，设为 `0` 可关闭。
- `no_infer_after_training`: 是否关闭训练结束后的最终推理。
- `infer_sample_index`: 从 `infer_manifest` 里取第几个样本做可视化。
- `infer_num_steps`: 可视化推理步数。
- `infer_seed`: 可视化随机种子。
- `infer_guidance`: 可视化 guidance；为 `null` 时使用 `guidance`。
- `infer_out`: 训练结束最终图像保存路径。周期性图像保存在 `output_dir/visualizations/step_xxxxxx.jpg`。

---

## 2. 推理/验证配置

推理入口：

```bash
python src/infer_flux_multiview.py --config configs/val/full_view_stage3_30_60_v4.json
```

推荐使用封装脚本：

```bash
GPU_IDS=0 bash scripts/infer_with_config.sh configs/val/full_view_stage3_30_60_v4.json
```

覆盖 config 参数：

```bash
GPU_IDS=0 bash scripts/infer_with_config.sh \
  configs/val/full_view_stage3_30_60_v4.json \
  --sample_index 3 \
  --out outputs/eval/stage3_sample3.jpg
```

### 推理配置列表

| 配置文件 | 用途 |
|---|---|
| `configs/val/debug_v2_100steps.json` | debug checkpoint 推理 |
| `configs/val/full_view_stage1_0_30_v4.json` | stage1 checkpoint 推理 |
| `configs/val/full_view_stage2_15_45_v4.json` | stage2 checkpoint 推理 |
| `configs/val/full_view_stage3_30_60_v4.json` | stage3 checkpoint 推理 |
| `configs/val/ablation_same_token_stage1_v4.json` | same_token 消融 checkpoint 推理 |

### 推理主要字段

- `mv_ckpt`: 已训练的 MVS adapter checkpoint。
- `manifest`: 验证 jsonl。若不用 manifest，也可以用 `camera_json`。
- `camera_json`: 手动指定相机外参数组，格式为 `[V,12]` 或 `{"extrinsics": [V,12]}`。
- `prompt`: 手动 prompt。使用 manifest 且样本中有 `prompt` 时可以省略。
- `sample_index`: 从 manifest 中取第几个样本。
- `num_views`: 推理视角数。
- `height`, `width`: 输出分辨率。
- `num_steps`: FLUX 采样步数。
- `guidance`: FLUX guidance scale。
- `seed`: 推理随机种子。
- `out`: 输出图片路径。
- `mv_arch`, `mv_adapter_dim`, `mv_attn_mode`, `inject_single_blocks`, `single_block_stride`, `no_mv_timestep_modulation`: 必须与训练时的 MVS 结构保持一致。
- `hf_download`: 是否允许通过 HuggingFace 自动下载权重。

---

## 3. 新建实验建议

新建训练实验：

```bash
cp configs/train/full_view_stage1_0_30_v4.json configs/train/my_experiment.json
```

新建对应验证配置：

```bash
cp configs/val/full_view_stage1_0_30_v4.json configs/val/my_experiment.json
```

两者至少保持一致：

```text
num_views
mv_arch
mv_adapter_dim
mv_attn_mode
inject_single_blocks
single_block_stride
no_mv_timestep_modulation
```

训练配置中修改：

```text
output_dir
train_manifest
infer_manifest
max_steps
resume_mv_ckpt
```

验证配置中修改：

```text
mv_ckpt
manifest
out
```
