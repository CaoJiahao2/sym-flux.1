# fullview_syncprobe: 2-view full_view synchronization probe configs
目标：只维持 2 张图同步，同时尽量保持 FLUX 原始画质。所有实验强制 `mv_attn_mode = "full_view"`，不再使用 `same_token`。
固定训练集：

```text
data/samples/stride_8_angle_0-30_v2_train_samples.jsonl
```
运行示例：

```bash
bash scripts/train.sh configs/train/fullview_syncprobe/fv_2v_adim512_singleS4_lr2e-5_pseudo10_steps1500.json
bash scripts/infer.sh configs/val/fullview_syncprobe/fv_2v_adim512_singleS4_lr2e-5_pseudo10_steps1500.json --sample_index 0
```
## 实验矩阵

| config | arch | dim | single injection | lr | pseudo | steps | 目的 |
|---|---:|---:|---|---:|---:|---:|---|
| `fv_2v_adim256_dblonly_lr2e-5_pseudo10_steps1200` | `adapter` | 256 | off | 2e-05 | 0.1 | 1200 | 保守基线：只在 double blocks 插 full_view，小容量、低 LR、低 pseudo，观察质量上限和基础同步。 |
| `fv_2v_adim512_dblonly_lr2e-5_pseudo10_steps1200` | `adapter` | 512 | off | 2e-05 | 0.1 | 1200 | 容量对照：double-only 下 adapter dim 从 256 提到 512，检查细节同步是否仅受容量限制。 |
| `fv_2v_adim512_singleS8_lr2e-5_pseudo10_steps1500` | `adapter` | 512 | stride=8 | 2e-05 | 0.1 | 1500 | 稀疏 single-block 注入：测试 single blocks 对局部细节同步的收益，同时限制对 FLUX 的扰动。 |
| `fv_2v_adim512_singleS4_lr2e-5_pseudo10_steps1500` | `adapter` | 512 | stride=4 | 2e-05 | 0.1 | 1500 | 中等 single-block 注入：主力配置，增强细节同步，观察质量下降幅度。 |
| `fv_2v_adim512_singleS2_lr1e-5_pseudo10_steps1800` | `adapter` | 512 | stride=2 | 1e-05 | 0.1 | 1800 | 高频 single-block 注入 + 低 LR：最大化细节同步，但用低 LR/大累积减轻质量损伤。 |
| `fv_2v_adim768_singleS4_lr1e-5_pseudo05_steps1800` | `adapter` | 768 | stride=4 | 1e-05 | 0.05 | 1800 | 更大容量 + 更低 pseudo：专门验证细节不同步是否来自同步通道容量不足/regularization 过强。 |
| `fv_2v_adim512_singleS4_notmod_lr2e-5_pseudo10_steps1500` | `adapter` | 512 | stride=4 | 2e-05 | 0.1 | 1500 | 关闭 timestep modulation：排查时间步调制是否导致不同 denoising 阶段细节漂移。 |
| `fv_2v_fullhidden_dblonly_lr1e-5_pseudo10_steps1000` | `full_hidden` | 512 | off | 1e-05 | 0.1 | 1000 | full_hidden attention：可从 FLUX img_attn 初始化，更接近 SynCamMaster 的 attention-init 思路；显存最重。 |

## 建议顺序

1. 先跑 `fv_2v_adim512_singleS4_lr2e-5_pseudo10_steps1500`，这是主力细节同步配置。
2. 再跑 `fv_2v_adim512_dblonly_lr2e-5_pseudo10_steps1200` 和 `fv_2v_adim512_singleS8_lr2e-5_pseudo10_steps1500`，判断 single-block 注入是否必要。
3. 如果细节仍不同步，跑 `singleS2` 或 `adim768_singleS4`；如果质量下降明显，回退到 `singleS8` 或 `dblonly`。
4. `fullhidden` 只在显存允许时跑，用于验证 attention 初始化路径是否优于 adapter。

## 命名规则

`fv_2v_<arch/dim>_<single策略>_lr<lr>_pseudo<pseudo>_steps<steps>`。其中 `fv` 表示 full_view，`2v` 表示两视角。
