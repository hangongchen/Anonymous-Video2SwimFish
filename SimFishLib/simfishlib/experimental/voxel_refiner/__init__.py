"""V2 voxel refiner: image-Qwen coarse voxel + video -> implicit occupancy -> high-res mesh.

The previous voxel-output design is replaced by:
  - a 3D U-Net feature backbone over the 64^3 coarse voxel + visual conditioning,
  - an implicit occupancy MLP head (trilinear feature sampling + posenc -> logit),
  - dense query grid + marching cubes for high-res mesh output at inference.
"""
