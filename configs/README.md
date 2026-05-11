# configs 使用说明

本项目采用 config-first 工作流。训练和推理的默认参数都写在 JSON 文件中，shell 脚本只负责读取 config 并启动 Python 程序。

```text
configs/
├── train/                         # 训练配置，给 scripts/train.sh 使用
│   ├── default.json
│   ├── full_view_stage1_0_30_v4.json
│   ├── full_view_stage2_15_45_v4.json
│   ├── full_view_stage3_30_60_v4.json
│   └── fullview2_probe/            # 2-view full_view 实验配置
├── val/                           # 推理配置，给 scripts/infer.sh 使用
│   ├── default.json
│   ├── full_view_stage1_0_30_v4.json
│   ├── full_view_stage2_15_45_v4.json
│   ├── full_view_stage3_30_60_v4.json
│   └── fullview2_probe/
└── fullview2_probe_README.md       # 8 组实验说明
```

---

## 1. 训练 config

训练命令：

```bash
bash scripts/train.sh configs/train/default.json
```

临时覆盖参数：

```bash
bash scripts/train.sh configs/train/default.json --max_steps 50 --infer_every 0
```

主要字段：

| 字段 | 含义 |
|---|---|
| `train_manifest` | 训练 jsonl 文件 |
| `infer_manifest` | 训练中可视化使用的 jsonl 文件 |
| `output_dir` | 实验输出目录；非空时优先使用 |
| `resolution` | 训练图像分辨率 |
| `num_views` | 训练视角数 |
| `batch_size` | DataLoader batch size |
| `max_steps` | optimizer steps 数量 |
| `learning_rate` | AdamW 学习率 |
| `grad_accum` | 梯度累积步数 |
| `mixed_precision` | `bf16`、`fp16` 或 `fp32` |
| `mv_arch` | `adapter` 或 `full_hidden` |
| `mv_adapter_dim` | adapter 瓶颈维度 |
| `mv_attn_mode` | `full_view` 或 `same_token` |
| `inject_single_blocks` | 是否注入 single stream blocks |
| `single_block_stride` | single blocks 注入间隔 |
| `pseudo_general_prob` | 单视角复制正则概率，用于保护 FLUX 画质 |
| `save_every` | checkpoint 保存间隔 |
| `infer_every` | 训练中可视化间隔；设为 0 关闭 |
| `resume_mv_ckpt` | 继续训练的 MVS checkpoint |

注意：`noise_share_ratio` 在当前训练代码中会被忽略，不作为有效实验变量。

---

## 2. 推理 config

推理命令：

```bash
bash scripts/infer.sh configs/val/default.json
```

临时覆盖参数：

```bash
bash scripts/infer.sh configs/val/default.json --sample_index 3 --out outputs/demo.jpg
```

主要字段：

| 字段 | 含义 |
|---|---|
| `mv_ckpt` | MVS checkpoint 路径 |
| `manifest` | 验证 jsonl 文件 |
| `sample_index` | 使用 manifest 中第几个样本 |
| `prompt` | 手动 prompt；为空时用 manifest 中 prompt |
| `camera_json` | 手动指定 camera extrinsics |
| `num_views` | 推理视角数 |
| `height`, `width` | 输出尺寸 |
| `num_steps` | FLUX 采样步数 |
| `guidance` | guidance scale |
| `seed` | 随机种子 |
| `out` | 输出图片路径 |

推理 config 中的 MVS 架构字段必须与训练时一致，否则 checkpoint 校验会报错：

```text
mv_arch
mv_adapter_dim
mv_attn_mode
inject_single_blocks
single_block_stride
no_mv_timestep_modulation
mv_dropout
```

---

## 3. 新建实验模板

复制训练配置：

```bash
cp configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json \
   configs/train/my_exp.json
```

复制推理配置：

```bash
cp configs/val/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json \
   configs/val/my_exp.json
```

至少修改：

```text
# train config
output_dir
train_manifest
infer_manifest
max_steps
learning_rate
pseudo_general_prob

# val config
mv_ckpt
manifest
out
```
