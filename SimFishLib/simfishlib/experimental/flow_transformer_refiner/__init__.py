"""Experimental flow-transformer geometry refiner (PhysX-Anything style).

Isolated from the V2 implicit-occupancy refiner. Pipeline:

  image-trained Qwen voxel64 + image/video condition + noise + timestep
    -> FlowTransformerRefiner (DiT-style transformer over 4^3 patches of a
       32^3 target field)
    -> rectified-flow velocity prediction
    -> ODE integration (Euler) from noise -> clean TSDF
    -> marching cubes -> mesh

Training target: truncated SDF at 32^3, derived from `asset.obj` -> binary
occupancy -> scipy distance transform -> clipped to [-1, 1].

See `scripts/experimental/README_flow_transformer_refiner.md` for usage.
"""

from simfishlib.experimental.flow_transformer_refiner.target import (
    PatchTokenizer,
    voxel_to_tsdf,
    mesh_to_tsdf,
    mesh_to_signed_distance,
    sample_query_points_and_sdf,
    downsample_voxel,
)
from simfishlib.experimental.flow_transformer_refiner.implicit_head import (
    FlowTransformerRefinerImplicit,
    ImplicitSDFDecoder,
    build_implicit_model,
)
from simfishlib.experimental.flow_transformer_refiner.flow import (
    sample_x_t,
    velocity_target,
    flow_matching_loss,
    euler_sample,
)
from simfishlib.experimental.flow_transformer_refiner.model import (
    FlowTransformerRefiner,
    build_model,
)
from simfishlib.experimental.flow_transformer_refiner.dataset import (
    FlowRefinerDataset,
    ImplicitFlowRefinerConfig,
    ImplicitFlowRefinerDataset,
    discover_samples,
    split_by_asset,
)
from simfishlib.experimental.flow_transformer_refiner.inference import (
    sample_implicit_volume,
    sample_tsdf_volume,
    tsdf_to_mesh,
)
from simfishlib.experimental.flow_transformer_refiner.io import (
    save_checkpoint,
    load_checkpoint,
    write_obj,
)

__all__ = [
    "PatchTokenizer",
    "voxel_to_tsdf",
    "mesh_to_tsdf",
    "mesh_to_signed_distance",
    "sample_query_points_and_sdf",
    "downsample_voxel",
    "sample_x_t",
    "velocity_target",
    "flow_matching_loss",
    "euler_sample",
    "FlowTransformerRefiner",
    "build_model",
    "FlowTransformerRefinerImplicit",
    "ImplicitSDFDecoder",
    "build_implicit_model",
    "FlowRefinerDataset",
    "discover_samples",
    "split_by_asset",
    "sample_tsdf_volume",
    "sample_implicit_volume",
    "tsdf_to_mesh",
    "ImplicitFlowRefinerConfig",
    "ImplicitFlowRefinerDataset",
    "save_checkpoint",
    "load_checkpoint",
    "write_obj",
]
