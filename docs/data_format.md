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
