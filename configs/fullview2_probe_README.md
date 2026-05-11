# fullview2_probe：2-view + full_view 实验组

目标：在只生成 2 张图的前提下，使用 `full_view` 提升跨视角同步，同时尽量减小相对原始 FLUX 的画质下降。

所有训练配置固定使用：

```text
data/samples/stride_8_angle_0-30_v2_train_samples.jsonl
```

所有配置均为：

```json
{"num_views": 2, "mv_attn_mode": "full_view"}
```

注意：当前代码中的 `noise_share_ratio` 已废弃并忽略，因此本实验组不把它作为变量。

## 实验列表

| 配置 | 目的 |
|---|---|
| `fullview2_adapter_dim128_lr5em5_p060_doubleOnly_steps800.json` | 最保守 baseline：小 adapter、只注入 double blocks、高 pseudo-general，用于确认画质上限。 |
| `fullview2_adapter_dim256_lr5em5_p060_doubleOnly_steps800.json` | 容量略增，判断 dim128 是否同步能力不足。 |
| `fullview2_adapter_dim256_lr3em5_p065_doubleOnly_steps1200.json` | 低学习率、更强正则、更长训练，判断画质下降是否来自优化过冲。 |
| `fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json` | 稀疏 single-block 注入，主要测试细节同步是否提升。 |
| `fullview2_adapter_dim256_lr5em5_p055_singleS4_steps1000.json` | 更密 single-block 注入，测试细节同步与画质下降的边界。 |
| `fullview2_adapter_dim384_lr3em5_p060_singleS8_do005_steps1200.json` | 更大 adapter + dropout + 低学习率，测试 dim256 是否欠拟合细节同步。 |
| `fullview2_adapter_dim256_lr5em5_p060_singleS8_noTmod_steps1000.json` | 关闭 timestep modulation，判断时步条件是否引入伪影或过拟合。 |
| `fullview2_fullhidden_lr2em5_p070_doubleOnly_steps600.json` | full_hidden 路线，显存压力更高，但更接近直接复用 FLUX hidden attention 的强同步方案。 |

## 建议顺序

1. 先跑 double-only 的前三组，确定画质上限和基本同步能力。
2. 如果细节不同步，再跑 `singleS8` 和 `singleS4`。
3. 如果仍欠拟合，跑 `dim384`。
4. 显存允许时跑 `fullhidden`。

## 命令示例

```bash
bash scripts/train.sh configs/train/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json
bash scripts/infer.sh configs/val/fullview2_probe/fullview2_adapter_dim256_lr5em5_p060_singleS8_steps1000.json --sample_index 0
```
