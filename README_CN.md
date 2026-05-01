# SynCamMaster-style FLUX 多视角文生图训练说明

本项目在 **FLUX.1-dev** 基础上加入 SynCamMaster 风格的 Multi-View Synchronization, MVS 模块，用于实现：

```text
输入：一个文本 prompt + V 个相机外参
输出：同一场景的 V 张多视角图像
```

当前默认实验配置：

```text
mv_attn_mode           = full_view
inject_single_blocks   = True
pseudo_general_prob    = 0.25
noise                  = 训练、推理均使用独立 Gaussian noise
角度 curriculum         = [0,30] -> [15,45] -> [30,60]
默认 num_views          = 4
默认 resolution         = 512
```

---

## 1. 代码结构

重要文件如下：

```text
src/models/multiview_sync.py          # MVS block：full_view / same_token, adapter / full_hidden
src/models/flux_multiview.py          # FluxMultiView：在 double/single blocks 插入 MVS
src/models/flux_multiview_loader.py   # 加载 FLUX.1-dev + MVS checkpoint
src/train_flux_multiview.py           # 训练脚本：日志、TensorBoard、训练后推理
src/infer_flux_multiview.py           # 多视角推理脚本
src/data/build_manifest.py            # 制作训练/验证 manifest
src/data/syncam_dataset.py            # SynCamVideo 多视角图像组 Dataset
src/data/camera_utils.py              # 相机外参解析与归一化

scripts/full_view/                    # 推荐使用的一键脚本
```

新增推荐脚本：

```text
scripts/full_view/00_check_forward_full_view.sh
scripts/full_view/05_make_scene_previews.sh
scripts/full_view/06_caption_scene_qwen.sh
scripts/full_view/10_build_manifests_full_view_angles.sh
scripts/full_view/20_train_full_view_0_30.sh
scripts/full_view/21_train_full_view_15_45.sh
scripts/full_view/22_train_full_view_30_60.sh
scripts/full_view/23_train_all_full_view_stages.sh
scripts/full_view/30_infer_full_view_latest.sh
scripts/full_view/40_tensorboard.sh
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

本项目没有依赖 `pip install -e .`，运行脚本时会自动设置：

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

如果你的路径不同，运行前设置：

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

如需允许从 Hugging Face 下载，可设置：

```bash
HF_DOWNLOAD=1 bash scripts/00_local_flux_env.sh
```

但默认脚本使用离线模式。

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

先预览一个 scene 的多视角图像：

```bash
DATASET_ROOT=data/SynCamVideo-Dataset \
SPLIT=train \
SCENE=scene10 \
FRAME_IDX=80 \
bash scripts/data_process/check_data.sh
```

输出图像默认在：

```text
outputs/preview_train_f24_aperture5_scene10_frame80_cam01-cam10.jpg
```

---

## 5. 可选但强烈建议：制作 scene-level captions

如果没有 captions，manifest 会使用 fallback prompt：

```text
a realistic 3D-rendered scene with a character performing an action
```

这可以跑通训练，但不利于 FLUX 学到文本与图像内容的对应关系。建议准备：

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

caption 原则：

```text
应该描述：主体、动作、场景、风格
不要描述：camera, view, left, right, front, back, angle, cam01 等视角词
```

### 5.1 制作 scene preview

```bash
DATASET_ROOT=data/SynCamVideo-Dataset \
bash scripts/full_view/05_make_scene_previews.sh
```

输出：

```text
data/scene_previews/train/
data/scene_previews/val/
```

### 5.2 使用本地 Qwen/VLM 生成 caption

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-9B \
GPU_ID=0 \
bash scripts/full_view/06_caption_scene_qwen.sh
```

生成后会自动调用 `src/caption/check_captions.py` 检查长度和禁用词。

如果你已有 captions，直接放到：

```text
data/captions/captions_scene_level_train.json
data/captions/captions_scene_level_val.json
```

即可。

---

## 6. 构建三个角度区间的 manifest

默认构建：

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

---

## 7. 检查 manifest 和 dataloader

例如检查 `[0,30]`：

```bash
MANIFEST=data/samples/stride_8_angle_0-30_v4_train_samples.jsonl \
RESOLUTION=512 \
NUM_VIEWS=4 \
bash scripts/data_process/check_manifest.sh
```

你应该看到类似：

```text
item pixel_values: (4, 3, 512, 512)
item cameras: (4, 12)
batch pixel_values: (1, 4, 3, 512, 512)
batch cameras: (1, 4, 12)
```

---

## 8. 功能检查：只跑模型 forward，不训练

先检查模型结构是否能正确前向传播：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
NUM_VIEWS=4 \
bash scripts/full_view/00_check_forward_full_view.sh
```

默认是：

```text
mv_attn_mode=full_view
inject_single_blocks=True
single_block_stride=4
pseudo_general_prob=0.25
```

测试 full hidden 版本：

```bash
GPU_IDS=0 \
MV_ARCH=full_hidden \
NUM_VIEWS=4 \
SEQ_LEN=1024 \
bash scripts/full_view/00_check_forward_full_view.sh
```

注意：`full_hidden` 显存消耗远高于 `adapter`。

---

## 9. 小步 smoke training

建议正式训练前先跑 10 step：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
MAX_STEPS=10 \
SAVE_EVERY=10 \
MAX_TRAIN_SCENES=50 \
MAX_VAL_SCENES=10 \
bash scripts/full_view/20_train_full_view_0_30.sh
```

检查输出目录：

```text
outputs/full_view_angle_0-30_adapter/
├── args.json
├── hparams.json
├── train.log
├── tensorboard/
├── mv_adapter_step_10.pt
├── mv_adapter_last.pt
├── train_state.json
└── final_inference_angle_0-30.jpg
```

必须确认：

```text
1. train.log 中 loss 不是 NaN
2. tensorboard 中 loss/step_raw 有曲线
3. final_inference_angle_0-30.jpg 成功保存
4. mv_adapter_last.pt 成功保存
```

查看 TensorBoard：

```bash
bash scripts/full_view/40_tensorboard.sh
```

浏览器打开：

```text
http://127.0.0.1:6006
```

---

## 10. 正式训练：三阶段角度 curriculum

### 10.1 阶段一：[0,30]

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
MAX_STEPS=5000 \
bash scripts/full_view/20_train_full_view_0_30.sh
```

默认输出：

```text
outputs/full_view_angle_0-30_adapter/
```

### 10.2 阶段二：[15,45]

默认会自动从上一阶段继续：

```text
outputs/full_view_angle_0-30_adapter/mv_adapter_last.pt
```

运行：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
MAX_STEPS=10000 \
bash scripts/full_view/21_train_full_view_15_45.sh
```

默认输出：

```text
outputs/full_view_angle_15-45_adapter/
```

### 10.3 阶段三：[30,60]

默认会自动从上一阶段继续：

```text
outputs/full_view_angle_15-45_adapter/mv_adapter_last.pt
```

运行：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
MAX_STEPS=15000 \
bash scripts/full_view/22_train_full_view_30_60.sh
```

默认输出：

```text
outputs/full_view_angle_30-60_adapter/
```

### 10.4 一键顺序训练三个阶段

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
bash scripts/full_view/23_train_all_full_view_stages.sh
```

---

## 11. full hidden 版本训练

低维 adapter 是默认推荐配置：

```bash
MV_ARCH=adapter
```

full hidden 版本：

```bash
MV_ARCH=full_hidden
```

它的特点：

```text
hidden_size = 3072
view_attn 使用 FLUX SelfAttention
初始化时从每个 double_block.img_attn 拷贝权重
更贴近 SynCamMaster 的初始化策略
显存占用显著更高
```

运行示例：

```bash
GPU_IDS=0 \
MV_ARCH=full_hidden \
GRAD_ACCUM=16 \
MAX_STEPS=1000 \
bash scripts/full_view/20_train_full_view_0_30.sh
```

如果显存不足，优先降低：

```text
1. RESOLUTION=384
2. NUM_VIEWS=2
3. BATCH_SIZE=1
4. SINGLE_BLOCK_STRIDE=8
5. MAX_STEPS 先用小步数检查
```

例如：

```bash
GPU_IDS=0 \
MV_ARCH=full_hidden \
RESOLUTION=384 \
NUM_VIEWS=2 \
GRAD_ACCUM=16 \
SINGLE_BLOCK_STRIDE=8 \
MAX_STEPS=100 \
bash scripts/full_view/20_train_full_view_0_30.sh
```

---

## 12. 训练日志、超参数和 loss 曲线

每次训练的 `OUTPUT_DIR` 中会保存：

```text
args.json             # 命令行参数
hparams.json          # 关键超参数与 trainable params
train.log             # 文本日志
tensorboard/          # TensorBoard events
train_state.json      # 最终 step、checkpoint 路径
mv_adapter_step_*.pt  # 中间 checkpoint
mv_adapter_last.pt    # 最终 checkpoint
final_inference*.jpg  # 训练结束后的自动推理结果
```

查看 loss：

```bash
LOGDIR=outputs bash scripts/full_view/40_tensorboard.sh
```

TensorBoard 中重点看：

```text
loss/step_raw
loss/running_avg
train/lr
train/grad_norm
train/pseudo_general_micro_batches
batch/used_pseudo_general
```

---

## 13. 手动推理验证

默认使用最终阶段 checkpoint：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
bash scripts/full_view/30_infer_full_view_latest.sh
```

默认读取：

```text
MV_CKPT=outputs/full_view_angle_30-60_adapter/mv_adapter_last.pt
MANIFEST=data/samples/stride_8_angle_30-60_v4_val_samples.jsonl
```

默认输出：

```text
outputs/full_view_angle_30-60_adapter/manual_inference.jpg
```

指定样本和输出：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
SAMPLE_INDEX=5 \
OUT=outputs/eval/sample_5.jpg \
bash scripts/full_view/30_infer_full_view_latest.sh
```

指定 checkpoint：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
MV_CKPT=outputs/full_view_angle_15-45_adapter/mv_adapter_last.pt \
MANIFEST=data/samples/stride_8_angle_15-45_v4_val_samples.jsonl \
OUT=outputs/eval/angle_15_45.jpg \
bash scripts/full_view/30_infer_full_view_latest.sh
```

---

## 14. 关键默认参数说明

### 14.1 full_view

`full_view` 会把同一 batch 内的多视角 image tokens 组织成：

```text
[B, V*S, D]
```

每个 view 的每个 spatial token 都可以 attend 到其他 view 的所有 spatial tokens。相比 `same_token`，它更适合中大视角差，但显存更高。

### 14.2 显式 spatial/view positional encoding

`MultiViewSyncBlock` 中加入了 `[view_id, y, x]` positional encoding：

```text
view_id: 归一化到 [0,1]
y, x:    来自 FLUX img_ids，并归一化到 [0,1]
```

这对 `full_view` 必要，否则 `V*S` 个 tokens 混合后缺少 view/spatial 位置信息。

### 14.3 inject_single_blocks=True

默认同时在：

```text
double-stream blocks 后
single-stream blocks 中的 image token 部分
```

插入 MVS。single-stream 中会先 split text/image tokens，只对 image tokens 做 MVS，不会对 text tokens 做跨视角同步。

### 14.4 pseudo_general_prob=0.25

训练时有 25% micro-batch 会执行：

```text
随机选一个 view 图像
复制成 V 个 views
所有 camera 设置为 identity [I|0]
```

目的：防止 MVS adapter 过度改变 FLUX 的原始视觉质量。

### 14.5 训练和推理均使用独立 noise

当前实现中训练和推理都使用每个 view 独立采样的 Gaussian noise。`--noise_share_ratio` 是兼容旧脚本的保留参数，不再生效。

---

## 15. 推荐评估方式

至少保存并对比以下结果：

```text
1. 原始 FLUX：同 prompt 独立生成 V 张
2. adapter + full_view + single MVS
3. full_hidden + full_view + single MVS
4. 不同角度阶段 checkpoint 的结果
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

## 16. 常见问题

### Q1：`train.log` 有 loss，但图像很不一致

优先检查：

```text
1. captions 是否太泛或错误
2. manifest 角度是否过大
3. 是否直接从 [30,60] 开始训练
4. 是否只训练了很少 steps
5. camera_extrinsics.json 是否正确解析
```

建议先只跑 `[0,30]`，确认小角度一致性。

### Q2：full_hidden OOM

优先改：

```bash
RESOLUTION=384
NUM_VIEWS=2
SINGLE_BLOCK_STRIDE=8
GRAD_ACCUM=16
```

仍 OOM 则先用 `MV_ARCH=adapter` 跑通主流程。

### Q3：TensorBoard 没有曲线

确认训练确实写入：

```text
outputs/.../tensorboard/events.out.tfevents...
```

然后：

```bash
LOGDIR=outputs bash scripts/full_view/40_tensorboard.sh
```

### Q4：训练后没有自动推理图

检查是否设置了：

```bash
NO_INFER_AFTER_TRAINING=1
```

或者查看 `train.log` 中 `Running end-of-training inference` 附近的错误信息。

---

## 17. 最小完整流程

第一次跑，建议使用小数据和小步数：

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
GPU_IDS=0 \
MV_ARCH=adapter \
bash scripts/full_view/00_check_forward_full_view.sh

# 5. smoke training
GPU_IDS=0 \
MV_ARCH=adapter \
MAX_STEPS=10 \
SAVE_EVERY=10 \
bash scripts/full_view/20_train_full_view_0_30.sh

# 6. 查看 loss
LOGDIR=outputs bash scripts/full_view/40_tensorboard.sh

# 7. 手动推理
GPU_IDS=0 \
MV_ARCH=adapter \
MV_CKPT=outputs/full_view_angle_0-30_adapter/mv_adapter_last.pt \
MANIFEST=data/samples/stride_8_angle_0-30_v4_val_samples.jsonl \
OUT=outputs/eval/smoke_infer.jpg \
bash scripts/full_view/30_infer_full_view_latest.sh
```

确认以上全部正常后，再运行完整三阶段：

```bash
GPU_IDS=0 \
MV_ARCH=adapter \
bash scripts/full_view/23_train_all_full_view_stages.sh
```
