# SynCamMaster-style FLUX 多视角文生图

本项目参考 SynCamMaster 的多视角同步思想，在 **FLUX.1-dev** 的 Transformer 上插入 Multi-View Synchronization（MVS）模块，实现：

```text
输入：一个文本 prompt + N 个相机外参
输出：同一动态/静态场景的 N 张同步多视角图像
```

当前工程重点面向 **文生多视角图像**，不是视频生成。训练数据来自 SynCamMaster 的 `SynCamVideo-Dataset`，但训练时按帧读取视频，把同一 scene、同一 frame、不同 camera 的画面组成一个多视角图像组。

---

## 0. 当前版本要点

- **训练和推理均为 config-first**：默认从 `configs/train/*.json` 和 `configs/val/*.json` 读取参数。
- **脚本已精简**：训练只用 `scripts/train.sh`，推理只用 `scripts/infer.sh`，manifest 构建只用 `scripts/build_manifest.sh`。
- **FLUX base model 在训练中冻结**：只训练新增的 MVS 相关参数。
- **支持两种 MVS 架构**：`adapter` 和 `full_hidden`。
- **支持两种跨视角 attention 模式**：`full_view` 和 `same_token`，当前推荐实验使用 `full_view`。
- **checkpoint 加载做架构强校验**：推理会读取 checkpoint 同级目录的 `args.json`，防止用错误架构加载权重。
- **训练脚本有空数据保护**：manifest 为空或 DataLoader 没有 batch 会直接报错，不会无限空跑。
- **注意**：当前 `noise_share_ratio` 参数在训练代码中会被打印为 deprecated 并忽略，训练和推理仍使用独立 Gaussian noise。不要把它作为有效实验变量。

---

## 1. 项目结构

```text
sym-flux/
├── configs/
│   ├── train/                         # 训练配置
│   │   ├── default.json
│   │   ├── full_view_stage*_*.json
│   │   └── fullview2_probe/            # 2-view full_view 实验配置
│   ├── val/                           # 推理/验证配置
│   │   ├── default.json
│   │   ├── full_view_stage*_*.json
│   │   └── fullview2_probe/
│   ├── README.md                      # config 字段说明
│   └── fullview2_probe_README.md       # 8 组 2-view 实验说明
│
├── scripts/
│   ├── 00_local_flux_env.sh            # 本地 FLUX 权重路径配置
│   ├── train.sh                        # config-first 训练入口
│   ├── infer.sh                        # config-first 推理入口
│   ├── build_manifest.sh               # manifest 构建入口
│   ├── check_forward.sh                # 前向检查
│   ├── tensorboard.sh                  # TensorBoard 启动
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── build_manifest.py           # 从 SynCamVideo-Dataset 构建 jsonl manifest
│   │   ├── syncam_dataset.py           # 多视角图像组 Dataset
│   │   └── camera_utils.py             # camera 外参解析与归一化
│   ├── models/
│   │   ├── multiview_sync.py           # MVS block
│   │   ├── flux_multiview.py           # 在 FLUX blocks 中插入 MVS
│   │   └── flux_multiview_loader.py    # FLUX + MVS checkpoint 加载与架构校验
│   ├── train_flux_multiview.py         # 训练主程序
│   ├── infer_flux_multiview.py         # 推理主程序
│   └── flux/                           # FLUX 相关代码
│
├── data/                               # 推荐放 manifest、caption、小样本数据
└── outputs/                            # 训练输出、checkpoint、可视化结果
```

---

## 2. 方法概览

### 2.1 基本流程

训练时，一个 batch 的数据形状可以理解为：

```text
pixel_values: [B, V, 3, H, W]
cameras:      [B, V, 12]
prompts:      List[str], length = B
```

其中：

- `B` 是 batch size；
- `V` 是视角数，例如 2 或 4；
- `cameras` 是展平后的相机外参，每个视角 12 维；
- 多视角图像来自同一个 scene、同一个 frame、不同 camera。

模型生成时把 `B*V` 张图放进 FLUX latent 流程，但 MVS 模块会按 `B,V` 重新组织特征，让不同 view 之间做 cross-view attention。

### 2.2 MVS 模块

核心模块在 `src/models/multiview_sync.py`。简化逻辑是：

```text
1. 将 [B*V, S, D] reshape 为 [B, V, S, D]
2. 将 camera extrinsics 编码为 camera embedding
3. 把 camera embedding 注入每个 view 的 token feature
4. 使用 view attention 在不同视角之间交换信息
5. projector 输出残差，加回原 FLUX feature
```

其中 `mv_attn_mode` 有两种：

| 模式 | 含义 | 适用情况 |
|---|---|---|
| `full_view` | 所有视角、所有 spatial tokens 联合 attention | 同步能力强，当前主要实验推荐 |
| `same_token` | 只让不同视角的同一 spatial index 通信 | 计算较轻，但同步能力较弱 |

### 2.3 MVS 架构

| 架构 | 参数 | 特点 |
|---|---|---|
| `adapter` | `mv_adapter_dim` 控制瓶颈维度 | 省显存，适合主要实验 |
| `full_hidden` | 直接在 FLUX hidden dimension 上做 view attention | 更重，显存压力更大，但同步能力可能更强 |

### 2.4 Base model 冻结策略

训练脚本会调用：

```python
model.freeze_base_model()
```

其策略是：

```text
1. 先冻结整个 FLUX + MVS model 的所有参数
2. 再只打开 mv_double_blocks / mv_single_blocks 的 requires_grad
3. optimizer 只接收 MVS trainable parameters
```

因此训练目标是：**保留 FLUX 原有文生图能力，只学习跨视角同步模块**。

---

## 3. 环境准备

建议使用 Python 3.10 或 3.11，CUDA 版本按本机 PyTorch 环境调整。

```bash
conda create -n flux_mv python=3.10 -y
conda activate flux_mv

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub safetensors sentencepiece protobuf accelerate
pip install decord opencv-python pillow einops tqdm numpy tensorboard
```

如果项目还有额外依赖，按报错补装即可。

---

## 4. 配置 FLUX.1-dev 本地权重

编辑：

```bash
scripts/00_local_flux_env.sh
```

默认示例：

```bash
export LOCAL_FLUX_DIR="${LOCAL_FLUX_DIR:-/data/model_cjh/FLUX.1-dev}"
export FLUX_DEV="${FLUX_DEV:-${LOCAL_FLUX_DIR}/flux1-dev.safetensors}"
export FLUX_MODEL="${FLUX_MODEL:-${FLUX_DEV}}"
export AE="${AE:-${LOCAL_FLUX_DIR}/ae.safetensors}"
export CLIP_L="${CLIP_L:-${LOCAL_FLUX_DIR}/clip_l.safetensors}"
export T5XXL="${T5XXL:-${LOCAL_FLUX_DIR}/t5xxl_fp16.safetensors}"
```

你的本地目录至少需要包含：

```text
FLUX.1-dev/
├── flux1-dev.safetensors
├── ae.safetensors
├── clip_l.safetensors
└── t5xxl_fp16.safetensors
```

也可以不改文件，直接在命令前设置环境变量：

```bash
export LOCAL_FLUX_DIR=/your/path/FLUX.1-dev
export FLUX_DEV=${LOCAL_FLUX_DIR}/flux1-dev.safetensors
export FLUX_MODEL=${FLUX_DEV}
export AE=${LOCAL_FLUX_DIR}/ae.safetensors
export CLIP_L=${LOCAL_FLUX_DIR}/clip_l.safetensors
export T5XXL=${LOCAL_FLUX_DIR}/t5xxl_fp16.safetensors
```

---

## 5. 准备 SynCamVideo-Dataset

推荐目录形式：

```text
SynCamVideo-Dataset/
├── train/
│   └── f24_aperture5/
│       └── scene*/
│           ├── videos/
│           │   ├── cam01.mp4
│           │   ├── cam02.mp4
│           │   └── ...
│           └── cameras/
│               └── camera_extrinsics.json
└── val/
    └── ...
```

构建 manifest 时需要指定 `DATASET_ROOT`：

```bash
export DATASET_ROOT=/your/path/SynCamVideo-Dataset
```

---

## 6. 构建 manifest

manifest 是训练/推理使用的 jsonl 索引文件。每一行对应一个多视角图像组，包含 scene、frame、videos、camera extrinsics、prompt 等信息。

通用入口：

```bash
DATASET_ROOT=/your/path/SynCamVideo-Dataset \
SPLIT=train \
OUT=data/samples/stride_8_angle_0-30_v2_train_samples.jsonl \
bash scripts/build_manifest.sh -- \
  --frame_stride 8 \
  --num_views 2 \
  --min_angle 0 \
  --max_angle 30 \
  --sampling random \
  --seed 1234
```

验证集示例：

```bash
DATASET_ROOT=/your/path/SynCamVideo-Dataset \
SPLIT=val \
OUT=data/samples/stride_8_angle_0-30_v2_val_samples.jsonl \
bash scripts/build_manifest.sh -- \
  --frame_stride 8 \
  --num_views 2 \
  --min_angle 0 \
  --max_angle 30 \
  --sampling random \
  --seed 1234
```

### 6.1 view 采样说明

`build_manifest.py` 支持：

```text
--sampling random
--sampling first
```

当前默认是 `random`。这样会从满足角度约束的 view 组合中随机采样，避免总是选到最小角度组合。

`first` 仅建议 debug 使用，不建议正式实验使用。

### 6.2 角度约束

常用配置：

```text
--min_angle 0 --max_angle 30
```

用于构建小视角差样本。若要更难的跨视角同步，可逐步扩大到：

```text
0-30
15-45
30-60
```

---

## 7. Config-first 训练

训练统一使用：

```bash
bash scripts/train.sh <train_config.json> [CLI overrides]
```

最小示例：

```bash
bash scripts/train.sh configs/train/default.json
```

覆盖 config 中的参数：

```bash
bash scripts/train.sh configs/train/default.json \
  --max_steps 50 \
  --save_every 10 \
  --infer_every 0
```

### 7.1 output_dir 规则

`scripts/train.sh` 的输出目录规则：

1. 如果 config 里有非空 `output_dir`，就使用 config 中的路径；
2. 如果 config 没有 `output_dir`，且 CLI 没有传 `--output_dir`，脚本按关键参数自动生成输出目录；
3. 如果 CLI 传了 `--output_dir`，CLI 优先。

### 7.2 训练输出

每个实验的 `output_dir` 下通常包含：

```text
outputs/xxx/
├── args.json                  # 训练参数快照；推理加载 checkpoint 时会读取并校验
├── mv_adapter_last.pt          # 最新 MVS checkpoint
├── mv_adapter_step_*.pt         # 周期保存的 checkpoint
├── events.out.tfevents.*       # TensorBoard 日志
└── visualizations/             # infer_every 生成的中间可视化图
```

---

## 8. Config-first 推理

推理统一使用：

```bash
bash scripts/infer.sh <val_config.json> [CLI overrides]
```

示例：

```bash
bash scripts/infer.sh configs/val/default.json
```

覆盖样本和输出路径：

```bash
bash scripts/infer.sh configs/val/default.json \
  --sample_index 3 \
  --out outputs/demo_sample3.jpg
```

手动指定 prompt：

```bash
bash scripts/infer.sh configs/val/default.json \
  --prompt "a young woman standing in a forest, cinematic, realistic" \
  --sample_index 0 \
  --out outputs/manual_prompt.jpg
```

### 8.1 checkpoint 架构校验

推理时 `mv_ckpt` 指向某个 MVS checkpoint，例如：

```json
"mv_ckpt": "outputs/fullview2_probe/xxx/mv_adapter_last.pt"
```

加载器会读取 checkpoint 同级目录中的：

```text
outputs/fullview2_probe/xxx/args.json
```

并校验当前推理配置和训练配置是否一致，包括：

```text
model_name
mv_arch
mv_adapter_dim
mv_attn_mode
inject_single_blocks
single_block_stride
mv_dropout
no_mv_timestep_modulation
```

如果不一致，会直接报错。这是为了避免用错误结构静默加载权重。

---

## 9. 推荐实验：2-view full_view 同步与画质探究

当前你主要关注两个问题：

```text
1. 两张图需要保持同步，尤其是人物、动作、服装、局部结构等细节。
2. 画质不能比原始 FLUX 下降太多。
```

因此新增了一组 **2-view + full_view** 实验配置：

```text
configs/train/fullview2_probe/
configs/val/fullview2_probe/
configs/fullview2_probe_README.md
```

所有训练配置固定使用：

```text
data/samples/stride_8_angle_0-30_v2_train_samples.jsonl
```

所有实验均满足：

```json
"num_views": 2,
"mv_attn_mode": "full_view"
```

### 9.1 8 组实验

| 实验配置 | 关键参数 | 目的 |
|---|---|---|
| `fullview2_adapter_dim128_lr5em5_p060_doubleOnly_steps800.json` | adapter, dim128, lr5e-5, pseudo0.60, double only | 画质优先 baseline |
| `fullview2_adapter_dim256_lr5em5_p060_doubleOnly_steps800.json` | adapter, dim256, lr5e-5, pseudo0.60, double only | 判断 dim128 是否同步能力不足 |
| `fullview2_adapter_dim256_lr3em5_p065_doubleOnly_steps1200.json` | adapter, dim256, lr3e-5, pseudo0.65, double only | 判断画质下降是否来自学习率过高 |
| `fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json` | adapter, dim256, single stride 8 | 测试稀疏 single-block 注入是否改善细节同步 |
| `fullview2_adapter_dim256_lr5em5_p055_singleS4_steps1000.json` | adapter, dim256, single stride 4 | 测试更密 single-block 注入的同步/画质边界 |
| `fullview2_adapter_dim384_lr3em5_p060_singleS8_do005_steps1200.json` | adapter, dim384, lr3e-5, dropout0.05 | 测试更大容量是否解决细节同步欠拟合 |
| `fullview2_adapter_dim256_lr5em5_p060_singleS8_noTmod_steps1000.json` | adapter, dim256, no timestep modulation | 测试 timestep modulation 是否引入伪影 |
| `fullview2_fullhidden_lr2em5_p070_doubleOnly_steps600.json` | full_hidden, lr2e-5, pseudo0.70 | 显存允许时测试更强同步结构 |

### 9.2 建议跑法

先跑这三组，判断画质和同步的基础 trade-off：

```bash
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim128_lr5em5_p060_doubleOnly_steps800.json
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_doubleOnly_steps800.json
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr3em5_p065_doubleOnly_steps1200.json
```

如果粗同步可以，但细节不同步，继续跑 single-block 组：

```bash
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p055_singleS4_steps1000.json
```

推理：

```bash
bash scripts/infer.sh configs/val/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json --sample_index 0
```

---

## 10. 关键 config 字段

### 10.1 数据字段

| 字段 | 含义 |
|---|---|
| `train_manifest` | 训练 jsonl 路径 |
| `infer_manifest` | 训练中可视化使用的 jsonl 路径 |
| `manifest` | 推理/验证 jsonl 路径 |
| `resolution` | 训练图像分辨率 |
| `height`, `width` | 推理输出尺寸 |
| `num_views` | 使用多少个视角 |

### 10.2 模型字段

| 字段 | 含义 |
|---|---|
| `mv_arch` | `adapter` 或 `full_hidden` |
| `mv_adapter_dim` | adapter 模式下的瓶颈维度 |
| `mv_attn_mode` | `full_view` 或 `same_token` |
| `inject_single_blocks` | 是否向 FLUX single stream blocks 注入 MVS |
| `single_block_stride` | single blocks 的注入间隔 |
| `no_mv_timestep_modulation` | 是否关闭 timestep modulation |
| `mv_dropout` | MVS dropout |

### 10.3 优化字段

| 字段 | 含义 |
|---|---|
| `max_steps` | optimizer steps，不是 micro steps |
| `learning_rate` | AdamW 学习率 |
| `weight_decay` | AdamW weight decay |
| `grad_accum` | 梯度累积步数 |
| `mixed_precision` | `bf16`、`fp16` 或 `fp32` |
| `gradient_checkpointing` | 是否开启 gradient checkpointing |

### 10.4 正则字段

| 字段 | 含义 |
|---|---|
| `pseudo_general_prob` | 以一定概率把单个 view 复制成多 view，保护 FLUX 原始画质 |
| `pseudo_general_random_view` | pseudo general 时是否随机选源视角 |
| `noise_share_ratio` | 当前代码中已废弃并忽略，不作为有效变量 |

### 10.5 保存与可视化字段

| 字段 | 含义 |
|---|---|
| `save_every` | 每多少 optimizer steps 保存一次 checkpoint |
| `infer_every` | 每多少 optimizer steps 做一次推理可视化；设为 0 关闭 |
| `no_infer_after_training` | 是否关闭训练结束后的最终推理 |
| `infer_num_steps` | 训练中可视化推理步数 |
| `infer_seed` | 训练中可视化随机种子 |
| `infer_out` | 最终推理图保存路径 |

---

## 11. 实验结果如何判断

建议每组实验至少固定 3 个验证样本、2 个 seed，对比下面四类现象。

### 11.1 同步性

观察：

```text
主体是否一致
动作姿态是否一致
服装颜色是否一致
脸部/头发/手部是否保持同一身份
场景布局是否符合相机视角变化
```

### 11.2 画质

观察：

```text
是否比原始 FLUX 明显糊
脸是否崩坏
树叶/纹理是否脏
光照是否异常灰暗
是否出现渲染数据集风格过拟合
```

### 11.3 视角控制

观察：

```text
两个 view 是否真的有相机角度差
是否只是复制近似同一张图
是否出现主体位置与相机不一致
```

### 11.4 过拟合倾向

观察：

```text
是否越来越像 SynCamVideo 的 UE 渲染风格
是否 prompt 多样性下降
是否背景或人物类别单一化
```

---

## 12. 常见问题

### Q1：生成图同步了，但画质明显下降

优先降低 MVS 对 FLUX 的干扰：

```text
降低 learning_rate
提高 pseudo_general_prob
减小 mv_adapter_dim
关闭或稀疏 single-block 注入
缩短 max_steps
```

优先尝试：

```bash
configs/train/fullview2_probe/fullview2_adapter_dim256_lr3em5_p065_doubleOnly_steps1200.json
```

### Q2：大体同步了，但细节不同步

说明 MVS 可能只学到语义级同步，没学到高频结构同步。优先尝试：

```text
打开 inject_single_blocks
使用 single_block_stride=8 或 4
适当增大 mv_adapter_dim
降低 pseudo_general_prob 一点
```

优先尝试：

```bash
configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json
configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p055_singleS4_steps1000.json
```

### Q3：推理时报 checkpoint 架构不匹配

原因是推理 config 和训练时 `args.json` 不一致。检查这些字段：

```text
mv_arch
mv_adapter_dim
mv_attn_mode
inject_single_blocks
single_block_stride
no_mv_timestep_modulation
mv_dropout
```

正确做法：复制对应的 `configs/val/...json`，不要手动混用其它实验的推理配置。

### Q4：训练一开始就报 manifest 为空

检查：

```text
train_manifest 路径是否存在
build_manifest.sh 的 DATASET_ROOT 是否正确
--split、--aperture 是否和数据目录一致
--min_angle/--max_angle 是否太严格，导致没有合法 view 组合
```

### Q5：训练非常慢或爆显存

优先调整：

```text
resolution: 512 -> 384
mv_arch: full_hidden -> adapter
mv_adapter_dim: 384/512 -> 128/256
inject_single_blocks: true -> false
single_block_stride: 4 -> 8
infer_every: 0
```

---

## 13. 最小完整流程

```bash
# 1. 配置权重
export LOCAL_FLUX_DIR=/your/path/FLUX.1-dev
export DATASET_ROOT=/your/path/SynCamVideo-Dataset

# 2. 构建 train manifest
DATASET_ROOT=$DATASET_ROOT \
SPLIT=train \
OUT=data/samples/stride_8_angle_0-30_v2_train_samples.jsonl \
bash scripts/build_manifest.sh -- \
  --frame_stride 8 --num_views 2 --min_angle 0 --max_angle 30 --sampling random

# 3. 构建 val manifest
DATASET_ROOT=$DATASET_ROOT \
SPLIT=val \
OUT=data/samples/stride_8_angle_0-30_v2_val_samples.jsonl \
bash scripts/build_manifest.sh -- \
  --frame_stride 8 --num_views 2 --min_angle 0 --max_angle 30 --sampling random

# 4. 训练一个推荐实验
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json

# 5. 推理验证
bash scripts/infer.sh configs/val/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json --sample_index 0

# 6. 看日志
bash scripts/tensorboard.sh outputs/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000
```

---

## 14. 开发约定

- 新实验优先新建 config，不要新增 stage-specific shell 脚本。
- 训练/推理必须优先从 config 读取参数；CLI 只用于临时覆盖。
- 每个 checkpoint 目录必须保留 `args.json`，否则推理加载会失败。
- 新增 MVS 架构参数时，要同步更新训练 config、推理 config 和 checkpoint 校验逻辑。
