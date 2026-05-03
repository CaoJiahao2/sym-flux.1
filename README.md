# SynCamMaster-style FLUX 多视角文生图训练说明

本项目在 **FLUX.1-dev** 的 Transformer 上加入 SynCamMaster 风格的 Multi-View Synchronization（MVS）模块，用于实现：

```text
输入：一个文本 prompt + V 个相机外参
输出：同一场景的 V 张多视角图像
```

当前版本重点更新：

```text
1. `configs/` 已拆分为 `configs/train/` 和 `configs/val/`。
2. 训练脚本支持 `--config configs/train/*.json`，命令行参数可以覆盖 config。
3. 推理脚本支持 `--config configs/val/*.json`，手动验证不再需要写长命令。
4. `infer_every` 默认每 500 optimizer steps 自动推理并保存一张多视角可视化图。
5. `scripts/full_view/` 下的训练与推理脚本均已简化为 config 驱动。
```

---

## 1. 代码结构

```text
configs/train/                        # 训练配置文件
configs/val/                          # 推理/验证配置文件
configs/README.md                     # 配置字段说明

src/config_utils.py                   # 读取 JSON/YAML config，校验字段
src/models/multiview_sync.py          # MVS block：full_view / same_token, adapter / full_hidden
src/models/flux_multiview.py          # FluxMultiView：在 double/single blocks 插入 MVS
src/models/flux_multiview_loader.py   # 加载 FLUX.1-dev + MVS checkpoint
src/train_flux_multiview.py           # 训练脚本：config、日志、TensorBoard、周期推理
src/infer_flux_multiview.py           # 多视角推理脚本
src/data/build_manifest.py            # 制作训练/验证 manifest
src/data/syncam_dataset.py            # SynCamVideo 多视角图像组 Dataset
src/data/camera_utils.py              # 相机外参解析与归一化

scripts/train_with_config.sh          # 通用 config 训练入口
scripts/infer_with_config.sh          # 通用 config 推理入口
scripts/full_view/                    # 推荐使用的一键脚本
```

---

## 2. 环境准备

建议使用 Python 3.10。

```bash
conda create -n flux_mv python=3.10 -y
conda activate flux_mv

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub safetensors sentencepiece protobuf accelerate
pip install decord opencv-python pillow einops tqdm numpy tensorboard
```

脚本会自动设置：

```bash
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"
```

---

## 3. 准备 FLUX.1-dev 本地权重

默认脚本读取：

```bash
LOCAL_FLUX_DIR=/data/model_cjh/FLUX.1-dev
```

该目录至少需要包含：

```text
/data/model_cjh/FLUX.1-dev/
├── flux1-dev.safetensors
├── ae.safetensors
├── text_encoder/
├── tokenizer/
├── text_encoder_2/
└── tokenizer_2/
```

如果路径不同，运行前设置：

```bash
export LOCAL_FLUX_DIR=/your/path/FLUX.1-dev
export FLUX_DEV=${LOCAL_FLUX_DIR}/flux1-dev.safetensors
export FLUX_MODEL=${FLUX_DEV}
export AE=${LOCAL_FLUX_DIR}/ae.safetensors
export FLUX_AE=${AE}
```

检查环境：

```bash
GPU_IDS=0 bash scripts/00_local_flux_env.sh
```

---

## 4. 准备 SynCamVideo-Dataset

推荐目录：

```text
data/SynCamVideo-Dataset/
├── train/
│   └── f24_aperture5/
│       └── scene*/
│           ├── videos/
│           │   ├── cam01.mp4
│           │   ├── ...
│           │   └── cam10.mp4
│           └── cameras/
│               └── camera_extrinsics.json
└── val/
    └── f24_aperture5/
        └── ...
```

如果数据不在默认目录，设置：

```bash
export DATASET_ROOT=/your/path/SynCamVideo-Dataset
```

预览一个 scene 的多视角图像：

```bash
DATASET_ROOT=data/SynCamVideo-Dataset \
SPLIT=train \
SCENE=scene10 \
FRAME_IDX=80 \
bash scripts/data_process/check_data.sh
```

---

## 5. 制作 scene-level captions

如果没有 captions，manifest 会使用 fallback prompt：

```text
a realistic 3D-rendered scene with a character performing an action
```

建议准备：

```text
data/captions/captions_scene_level_train.json
data/captions/captions_scene_level_val.json
```

格式：

```json
{
  "scene1": "a woman dancing in a modern room",
  "scene2": "a man walking on a city street"
}
```

caption 应描述主体、动作、场景、风格；不要出现 camera、view、left、right、front、back、angle、cam01 等视角词。

制作 scene preview：

```bash
DATASET_ROOT=data/SynCamVideo-Dataset \
bash scripts/full_view/05_make_scene_previews.sh
```

使用本地 Qwen/VLM 生成 caption：

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-9B \
GPU_ID=0 \
bash scripts/full_view/06_caption_scene_qwen.sh
```

---

## 6. 构建 manifest

默认构建三个角度区间：

```text
[0,30]
[15,45]
[30,60]
```

运行：

```bash
DATASET_ROOT=data/SynCamVideo-Dataset \
NUM_VIEWS=4 \
FRAME_STRIDE=8 \
SAMPLING=random \
bash scripts/full_view/10_build_manifests_full_view_angles.sh
```

默认输出：

```text
data/samples/stride_8_angle_0-30_v4_train_samples.jsonl
data/samples/stride_8_angle_0-30_v4_val_samples.jsonl

data/samples/stride_8_angle_15-45_v4_train_samples.jsonl
data/samples/stride_8_angle_15-45_v4_val_samples.jsonl

data/samples/stride_8_angle_30-60_v4_train_samples.jsonl
data/samples/stride_8_angle_30-60_v4_val_samples.jsonl
```

调试时可以限制场景数量：

```bash
DATASET_ROOT=data/SynCamVideo-Dataset \
NUM_VIEWS=4 \
MAX_TRAIN_SCENES=50 \
MAX_VAL_SCENES=10 \
bash scripts/full_view/10_build_manifests_full_view_angles.sh
```

检查 dataloader：

```bash
MANIFEST=data/samples/stride_8_angle_0-30_v4_train_samples.jsonl \
RESOLUTION=512 \
NUM_VIEWS=4 \
bash scripts/data_process/check_manifest.sh
```

---

## 7. Config 目录结构

现在 `configs/` 分为两个子目录：

```text
configs/train/    # 训练配置，供 train_flux_multiview.py 使用
configs/val/      # 推理/验证配置，供 infer_flux_multiview.py 使用
```

推荐配置：

```text
configs/train/debug_v2_100steps.json              # 2 视角、100 step 冒烟测试
configs/train/full_view_stage1_0_30_v4.json       # 4 视角，[0,30]
configs/train/full_view_stage2_15_45_v4.json      # 4 视角，[15,45]，从 stage1 继续
configs/train/full_view_stage3_30_60_v4.json      # 4 视角，[30,60]，从 stage2 继续
configs/train/ablation_same_token_stage1_v4.json  # same_token 消融实验

configs/val/debug_v2_100steps.json                # debug checkpoint 推理
configs/val/full_view_stage1_0_30_v4.json         # stage1 checkpoint 推理
configs/val/full_view_stage2_15_45_v4.json        # stage2 checkpoint 推理
configs/val/full_view_stage3_30_60_v4.json        # stage3 checkpoint 推理
configs/val/ablation_same_token_stage1_v4.json    # 消融 checkpoint 推理
```

完整字段说明见：

```text
configs/README.md
```

---

## 8. Config 驱动训练

使用封装脚本：

```bash
GPU_IDS=0 bash scripts/train_with_config.sh configs/train/full_view_stage1_0_30_v4.json
```

命令行参数会覆盖 config 文件中的同名字段：

```bash
GPU_IDS=0 bash scripts/train_with_config.sh \
  configs/train/full_view_stage1_0_30_v4.json \
  --max_steps 2000 \
  --infer_every 250
```

---

## 9. Config 驱动推理/验证

推理入口：

```bash
python src/infer_flux_multiview.py --config configs/val/full_view_stage3_30_60_v4.json
```

或使用封装脚本：

```bash
GPU_IDS=0 bash scripts/infer_with_config.sh configs/val/full_view_stage3_30_60_v4.json
```

命令行参数同样可以覆盖 config，例如换验证样本和输出路径：

```bash
GPU_IDS=0 bash scripts/infer_with_config.sh \
  configs/val/full_view_stage3_30_60_v4.json \
  --sample_index 3 \
  --out outputs/eval/stage3_sample3.jpg
```

`configs/val/*.json` 主要字段：

```text
mv_ckpt, manifest, sample_index, num_views, height, width, num_steps, guidance, seed, out
mv_arch, mv_adapter_dim, mv_attn_mode, inject_single_blocks, single_block_stride
```

---

## 10. 周期性推理可视化

新增参数：

```text
infer_every = 500
```

含义：每训练 `infer_every` 个 optimizer steps，自动从训练 config 的 `infer_manifest` 取 `infer_sample_index` 对应样本，生成一张多视角 grid 图。

默认保存位置：

```text
output_dir/visualizations/step_000500.jpg
output_dir/visualizations/step_001000.jpg
...
```

同时会写入 TensorBoard：

```text
infer/periodic_grid
```

关闭周期性推理：

```bash
python src/train_flux_multiview.py \
  --config configs/train/full_view_stage1_0_30_v4.json \
  --infer_every 0
```

训练结束后的最终推理仍然保留，输出到训练 config 中的 `infer_out`。如需关闭最终推理：

```bash
python src/train_flux_multiview.py \
  --config configs/train/full_view_stage1_0_30_v4.json \
  --no_infer_after_training
```

---

## 11. 正式训练：三阶段角度 curriculum

### 10.1 阶段一：[0,30]

```bash
GPU_IDS=0 bash scripts/full_view/20_train_full_view_0_30.sh
```

等价于：

```bash
GPU_IDS=0 bash scripts/train_with_config.sh configs/train/full_view_stage1_0_30_v4.json
```

默认输出：

```text
outputs/full_view_stage1_0_30_v4/
```

### 10.2 阶段二：[15,45]

```bash
GPU_IDS=0 bash scripts/full_view/21_train_full_view_15_45.sh
```

默认从以下 checkpoint 继续：

```text
outputs/full_view_stage1_0_30_v4/mv_adapter_last.pt
```

默认输出：

```text
outputs/full_view_stage2_15_45_v4/
```

### 10.3 阶段三：[30,60]

```bash
GPU_IDS=0 bash scripts/full_view/22_train_full_view_30_60.sh
```

默认从以下 checkpoint 继续：

```text
outputs/full_view_stage2_15_45_v4/mv_adapter_last.pt
```

默认输出：

```text
outputs/full_view_stage3_30_60_v4/
```

### 10.4 一键顺序训练

```bash
GPU_IDS=0 bash scripts/full_view/23_train_all_full_view_stages.sh
```

额外覆盖参数会传入三个阶段：

```bash
GPU_IDS=0 bash scripts/full_view/23_train_all_full_view_stages.sh \
  --infer_every 1000 \
  --infer_num_steps 20
```

---

## 12. 训练参数如何配置

现在主要参数都应该写在 config 中，而不是写在 shell 脚本里。重点参数如下：

```text
数据：train_manifest, infer_manifest, resolution, num_views, batch_size, num_workers
模型：model_name, mv_arch, mv_adapter_dim, mv_attn_mode, inject_single_blocks, single_block_stride
优化：max_steps, learning_rate, weight_decay, grad_accum, mixed_precision, gradient_checkpointing
正则：pseudo_general_prob, pseudo_general_random_view
日志：save_every, log_every, seed
推理：infer_every, infer_sample_index, infer_num_steps, infer_seed, infer_guidance, infer_out
恢复：resume_mv_ckpt
```

如果要新建实验，直接复制一个 config：

```bash
cp configs/train/full_view_stage1_0_30_v4.json configs/train/my_experiment.json
```

然后修改：

```text
output_dir
train_manifest
infer_manifest
max_steps
mv_attn_mode
mv_arch
single_block_stride
infer_every
```

运行：

```bash
GPU_IDS=0 bash scripts/train_with_config.sh configs/train/my_experiment.json
```

---

## 13. full_hidden 版本训练

默认推荐：

```json
"mv_arch": "adapter"
```

full hidden 版本：

```json
"mv_arch": "full_hidden"
```

它的特点：

```text
hidden_size = 3072
view_attn 使用 FLUX SelfAttention
初始化时从每个 double_block.img_attn 拷贝权重
更接近 SynCamMaster 的 view-attention 初始化思想
显存占用显著更高
```

建议先复制 config 再改：

```bash
cp configs/train/full_view_stage1_0_30_v4.json configs/train/full_hidden_stage1_0_30_v4.json
```

修改：

```json
{
  "mv_arch": "full_hidden",
  "output_dir": "outputs/full_hidden_stage1_0_30_v4",
  "grad_accum": 16,
  "single_block_stride": 8
}
```

如果显存不足，优先降低：

```text
1. resolution: 384
2. num_views: 2
3. batch_size: 1
4. single_block_stride: 8 或更大
5. infer_num_steps: 10，用于调试
```

---

## 14. 训练日志与输出

每次训练的 `output_dir` 中会保存：

```text
args.json                  # 最终生效参数，包括 config 合并结果
hparams.json               # 同 args.json，便于 TensorBoard 记录
train.log                  # 文本日志
config_snapshot.json        # 可选：如自行添加也可保存原始 config
tensorboard/               # TensorBoard events
visualizations/            # 周期性推理图 step_*.jpg
train_state.json           # 最终 step、checkpoint 路径
mv_adapter_step_*.pt       # 中间 checkpoint
mv_adapter_last.pt         # 最终 checkpoint
final_inference.jpg        # 训练结束后的最终推理结果
```

TensorBoard 中重点看：

```text
loss/step_raw
loss/running_avg
train/lr
train/grad_norm
train/pseudo_general_micro_batches
batch/used_pseudo_general
infer/periodic_grid
```

---

## 15. 手动推理验证

默认使用最终阶段 validation config：

```bash
GPU_IDS=0 bash scripts/full_view/30_infer_full_view_latest.sh
```

指定其他阶段：

```bash
GPU_IDS=0 bash scripts/infer_with_config.sh configs/val/full_view_stage1_0_30_v4.json
```

临时覆盖 checkpoint、样本编号或输出路径：

```bash
GPU_IDS=0 bash scripts/infer_with_config.sh \
  configs/val/full_view_stage1_0_30_v4.json \
  --mv_ckpt outputs/full_view_stage1_0_30_v4/mv_adapter_last.pt \
  --sample_index 2 \
  --out outputs/eval/stage1_sample2.jpg
```

---

## 16. 推荐消融实验

至少比较：

```text
1. 原始 FLUX：同 prompt 独立生成 V 张
2. adapter + full_view + single blocks
3. adapter + same_token + single blocks
4. full_hidden + full_view + single blocks
5. 不同角度阶段 checkpoint 的结果
```

重点观察：

```text
主体一致性：人物/动物身份、衣服、颜色是否一致
场景一致性：背景布局是否一致
视角响应：相机角度变化是否体现在图像中
文本对齐：是否仍符合 prompt
伪影：是否出现重复人脸、主体断裂、背景错位
```

---

## 17. 常见问题

### Q1：`train.log` 有 loss，但多视角图像很不一致

优先检查：

```text
1. captions 是否太泛或错误
2. manifest 角度是否过大
3. 是否直接从 [30,60] 开始训练
4. 训练 steps 是否太少
5. camera_extrinsics.json 是否正确解析
```

建议先只跑 `[0,30]`，确认小角度一致性。

### Q2：周期性推理太慢

改 config：

```json
"infer_every": 1000,
"infer_num_steps": 10
```

或者运行时覆盖：

```bash
bash scripts/full_view/20_train_full_view_0_30.sh --infer_every 1000 --infer_num_steps 10
```

### Q3：不想训练时推理

关闭周期性推理：

```bash
--infer_every 0
```

关闭最终推理：

```bash
--no_infer_after_training
```

### Q4：full_hidden OOM

优先改 config：

```json
"resolution": 384,
"num_views": 2,
"single_block_stride": 8,
"grad_accum": 16
```

仍 OOM 则先用 `mv_arch=adapter` 跑通主流程。

---

## 18. 最小完整流程

```bash
# 1. 数据预览
DATASET_ROOT=data/SynCamVideo-Dataset \
SPLIT=train \
SCENE=scene10 \
FRAME_IDX=80 \
bash scripts/data_process/check_data.sh

# 2. 构建小规模 manifest
DATASET_ROOT=data/SynCamVideo-Dataset \
NUM_VIEWS=4 \
MAX_TRAIN_SCENES=50 \
MAX_VAL_SCENES=10 \
bash scripts/full_view/10_build_manifests_full_view_angles.sh

# 3. 检查 dataloader
MANIFEST=data/samples/stride_8_angle_0-30_v4_train_samples.jsonl \
RESOLUTION=512 \
NUM_VIEWS=4 \
bash scripts/data_process/check_manifest.sh

# 4. 检查模型 forward
GPU_IDS=0 bash scripts/full_view/00_check_forward_full_view.sh

# 5. smoke training
GPU_IDS=0 bash scripts/train_with_config.sh \
  configs/train/debug_v2_100steps.json \
  --max_steps 20 \
  --save_every 10 \
  --infer_every 10

# 6. 查看 loss 和周期性推理图
LOGDIR=outputs bash scripts/full_view/40_tensorboard.sh

# 7. 正式三阶段训练
GPU_IDS=0 bash scripts/full_view/23_train_all_full_view_stages.sh
```
