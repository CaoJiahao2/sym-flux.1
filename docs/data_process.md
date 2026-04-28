manifest 单行格式,生成的每一行大致是：
{
  "dataset_root": ".../data/SynCamVideo-Dataset",
  "split": "train",
  "aperture": "f24_aperture5",
  "scene": "scene1",
  "frame_idx": 40,
  "frame_key": "frame40",
  "cams": ["cam01", "cam02", "...", "cam10"],
  "videos": [".../cam01.mp4", ".../cam02.mp4"],
  "extrinsics": [[12 floats], [12 floats]],
  "extrinsics_convention": "w2c",
  "anchor_cam": "cam01",
  "max_pairwise_rotation_deg": 123.45,
  "prompt": "a realistic 3D-rendered scene with a character performing an action"
}

新增文件如下：
src/
├── data/
│   └── syncam_dataset.py              # 支持 num_views 读取
├── models/
│   ├── multiview_sync.py              # Cross-view attention adapter
│   ├── flux_multiview.py              # 继承官方 Flux 的多视角模型
│   └── flux_multiview_loader.py       # 加载 FLUX.1-dev + adapter
├── training/
│   └── flux_train_utils.py            # latent pack、prompt encode、timestep
├── train_flux_multiview.py            # 训练脚本
├── infer_flux_multiview.py            # 推理脚本
└── check_flux_mv_forward.py           # forward 检查脚本

scripts/
├── 10_install_flux_official.sh
├── 11_check_flux_mv_forward.sh
├── 20_train_flux_mvs_stage1.sh
├── 21_train_flux_mvs_stage2.sh
└── 30_infer_flux_mvs_manifest.sh