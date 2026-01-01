"""Target representation for the flow-transformer refiner.

The refiner predicts a continuous **truncated SDF** (TSDF) at a small dense
resolution (default 64^3, matching the Qwen voxel input). We use TSDF rather
than binary occupancy because flow matching needs a continuous-valued field
with meaningful gradients.

Pipeline:
  asset.obj  -> normalized voxel R^3 (R=64 default)
             -> TSDF in voxel units (negative inside, positive outside)
             -> clipped to [-1, 1] via division by truncation distance
             -> patchified into (R/P)^3 spatial tokens of dim P^3 (default P=4)

Sign convention: **negative inside, positive outside**, so the surface is at
iso = 0 (matches marching_cubes default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def voxel_to_tsdf(occupancy: np.ndarray, truncation: float = 4.0) -> np.ndarray:
    """Convert a boolean occupancy grid to a normalized TSDF in [-1, 1].

    occupancy: bool or numeric array of shape [R, R, R] (non-zero == inside)
    truncation: clamp distance in voxel units. Distances are normalized by it
                so the returned field lives in [-1, 1].

    Returns float32 array of shape [R, R, R], negative inside / positive
    outside / 0 at the surface.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:
        raise RuntimeError("voxel_to_tsdf requires scipy") from exc
    occ = np.asarray(occupancy).astype(bool)
    if occ.ndim != 3:
        raise ValueError(f"occupancy must be 3D, got {occ.shape}")
    d_out = distance_transform_edt(~occ).astype(np.float32)
    if occ.any():
        d_in = distance_transform_edt(occ).astype(np.float32)
    else:
        d_in = np.zeros_like(d_out)
    sdf = d_out - d_in
    return np.clip(sdf / max(truncation, 1e-6), -1.0, 1.0).astype(np.float32)


def mesh_to_tsdf(
    mesh_path: Path, resolution: int = 32, truncation: float = 4.0
) -> np.ndarray:
    """Rasterize asset.obj to occupancy at `resolution`^3 then convert to TSDF.

    Quantized/staircase TSDF -- prefer ``mesh_to_signed_distance`` for training
    supervision; this remains as a cheap fallback.
    """
    from simfishlib.experimental.voxel_refiner.mesh_target import mesh_to_occupancy

    occ = mesh_to_occupancy(Path(mesh_path), resolution=resolution)
    return voxel_to_tsdf(occ, truncation=truncation)


def _extrude_shell(c: "trimesh.Trimesh", thickness: float) -> "trimesh.Trimesh":
    """Turn an open sheet into a closed solid of the given thickness by extruding
    its faces +/-0.5*thickness along the vertex normals and capping the open
    boundary with walls."""
    import trimesh as _tm

    V = np.asarray(c.vertices, dtype=np.float64)
    F = np.asarray(c.faces, dtype=np.int64)
    N = np.asarray(c.vertex_normals, dtype=np.float64)
    n = len(V)
    V_up = V + 0.5 * float(thickness) * N
    V_dn = V - 0.5 * float(thickness) * N
    F_up = F.copy()
    F_dn = F[:, ::-1].copy() + n  # flipped winding for outward-facing bottom
    # boundary edges = edges that appear in exactly one face
    edges_oriented = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges_sorted = np.sort(edges_oriented, axis=1)
    _, inverse, counts = np.unique(
        edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    boundary_mask = counts[inverse] == 1
    bd = edges_oriented[boundary_mask]
    walls = np.empty((len(bd) * 2, 3), dtype=np.int64)
    walls[0::2] = np.stack([bd[:, 0], bd[:, 1], bd[:, 1] + n], axis=1)
    walls[1::2] = np.stack([bd[:, 0], bd[:, 1] + n, bd[:, 0] + n], axis=1)
    V_all = np.vstack([V_up, V_dn])
    F_all = np.vstack([F_up, F_dn, walls])
    return _tm.Trimesh(V_all, F_all, process=False)


def _min_thickness_pca(c: "trimesh.Trimesh") -> float:
    """Orientation-invariant thickness: the extent along the component's smallest
    principal axis. Unlike the axis-aligned bounding box (`min(c.extents)`), this
    does not over-estimate the thickness of a tilted/curved flat sheet."""
    V = np.asarray(c.vertices, dtype=np.float64)
    if len(V) < 3:
        return 0.0
    Vc = V - V.mean(axis=0, keepdims=True)
    try:
        # principal axes = right-singular vectors; projected extents along each
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
        proj = Vc @ Vt.T
        return float(np.ptp(proj, axis=0).min())
    except np.linalg.LinAlgError:
        return float(min(c.extents))


def solidify_thin_components(
    mesh: "trimesh.Trimesh",
    thickness: float,
    thin_threshold: float,
) -> "trimesh.Trimesh":
    """Extrude paper-thin *open* sub-sheets of a mesh into closed solids so they
    survive voxelization / SDF sampling.

    Why fins disappear: paper-thin fins/tails are single layers of triangles with
    no enclosed volume (open surfaces). `igl.signed_distance` decides inside/outside
    by the winding number, which never reaches 1 for an open sheet, so the fin is
    treated as having no interior and is silently dropped -- at ANY resolution.
    Thickening rebuilds them as closed shells with a real interior.

    Selection criterion (per connected component):
      - We thicken a component only if it is **open** (not watertight) AND its true
        (orientation-invariant, PCA) thickness is below `thin_threshold`.
      - **Watertight** (closed) components are left untouched: they already enclose
        a volume so the SDF resolves them, and extruding a closed body would turn a
        solid into a hollow double-walled shell (its interior would read as outside).

    vs the previous version this:
      - uses PCA thickness instead of the axis-aligned bbox, so tilted/curved thin
        fins (whose bbox min-extent over-estimates thickness) are no longer missed;
      - gates on watertightness, so the body is provably safe and any open sheet of
        the right thickness is caught regardless of its bbox orientation.

    Note: a fin that is vertex-welded into the *same connected component* as the
    body is not isolated by `split` and therefore still won't be caught here -- that
    needs face-level surgery (a separate, heavier change).
    """
    import trimesh as _tm

    comps = mesh.split(only_watertight=False)
    out: list[_tm.Trimesh] = []
    for c in comps:
        is_open = not bool(c.is_watertight)
        is_thin = _min_thickness_pca(c) < float(thin_threshold)
        if is_open and is_thin:
            out.append(_extrude_shell(c, thickness))
        else:
            out.append(c)
    return _tm.util.concatenate(out)


# Alpha-wrap detail: alpha = bbox_diag / _ALPHA_WRAP_ALPHA_DIV. Smaller alpha =
# finer detail (resolves narrower gaps between fins); too small bridges nothing
# but is slower. diag/200 keeps fin gaps open while staying fast.
_ALPHA_WRAP_ALPHA_DIV = 200.0


def alpha_wrap_solidify(mesh: "trimesh.Trimesh", offset: float) -> "trimesh.Trimesh":
    """Wrap a watertight surface around ``mesh`` using CGAL 3D Alpha Wrapping.

    Unlike ``solidify_thin_components`` (per-connected-component, can't reach a
    thin tail welded into the body), alpha wrapping operates **globally** on the
    whole triangle soup: every open/thin sheet becomes a closed shell of thickness
    ~2*offset, and fins fused into the body are handled too. The output is always
    watertight, so ``igl.signed_distance`` resolves it correctly.

    ``offset`` is the distance the wrap stays from the input surface (so a thin
    fin ends up ~2*offset thick). ``alpha`` (detail) is derived from the bbox
    diagonal.

    Falls back to ``solidify_thin_components`` if CGAL is unavailable.
    """
    import trimesh as _tm

    try:
        from CGAL import CGAL_Alpha_wrap_3 as _aw
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    except Exception as exc:  # pragma: no cover - depends on optional dep
        import warnings
        warnings.warn(
            f"CGAL alpha wrap unavailable ({exc}); falling back to component "
            f"extrusion. `pip install cgal` to enable.",
            RuntimeWarning,
        )
        return solidify_thin_components(mesh, 2.0 * float(offset), 0.04)

    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    diag = float(np.linalg.norm(mesh.extents)) or 1.0
    alpha = diag / _ALPHA_WRAP_ALPHA_DIV

    pts = _aw.Point_3_Vector()
    pts.reserve(len(V))
    for p in V:
        pts.append(Point_3(float(p[0]), float(p[1]), float(p[2])))
    polys = _aw.Polygon_Vector()
    for f in F:
        iv = _aw.Int_Vector()
        for idx in f:
            iv.append(int(idx))
        polys.append(iv)
    out = Polyhedron_3()
    _aw.alpha_wrap_3(pts, polys, float(alpha), float(offset), out)

    import tempfile
    off = tempfile.mktemp(suffix=".off")
    out.write_to_file(off)
    wrapped = _tm.load(off, process=False)
    Path(off).unlink(missing_ok=True)
    return wrapped


def _load_normalized_mesh_maybe_thickened(
    mesh_path: Path,
    thicken: bool,
    fin_thickness: float,
    thin_threshold: float,
):
    """Load + normalize the mesh; optionally solidify thin sheets.

    When ``thicken`` is set we run CGAL **alpha wrapping** (global, robust to
    fins fused into the body) with shell thickness driven by ``fin_thickness``
    (offset = fin_thickness / 2, so the wrapped fin is ~fin_thickness thick).
    ``thin_threshold`` is unused by the alpha-wrap path (it is global, not
    per-component) but kept in the signature for cache-key compatibility.

    Normalization happens first so thickness/offset are in unit-cube coords.
    """
    from simfishlib.experimental.voxel_refiner.mesh_target import load_normalized_mesh

    mesh = load_normalized_mesh(Path(mesh_path))
    if thicken:
        mesh = alpha_wrap_solidify(mesh, offset=0.5 * float(fin_thickness))
    return mesh


# Per-point sampling mixture (fractions of the total pool). Concentrates training
# points near the surface -- especially thin parts -- instead of wasting capacity
# on trivially-easy far inside/outside points.
_SAMPLE_FRAC = {"near": 0.50, "hard": 0.10, "inside": 0.15, "outside": 0.25}

# Meshes are normalized so the longest axis fills [0,1] exactly -- head/tail tips
# sit on the cube boundary. We sample (and clip) over a slightly larger range so
# the model gets "outside" supervision *beyond* those boundary tips and learns
# where the fish actually ends, instead of bluntly filling to the wall.
_SAMPLE_MARGIN = 0.06


def _sample_query_points(
    mesh: "trimesh.Trimesh",
    total: int,
    sigma: float,
    rng: "np.random.Generator",
    seed: int,
) -> np.ndarray:
    """Sample a 4-way mixture of query points in [0,1]^3:

      50% near-surface : surf + n * N(0, sigma)        (the band where SDF ~ 0)
      10% hard         : surf + n * N(0, sigma/4)       (tight surface, thin parts)
      15% inside       : surf - n * |N(0, 1.5*sigma)|   (pushed inward, SDF < 0)
      25% outside      : uniform in the cube            (global far context)

    Normal-offset sampling for inside/outside is reliable because the GT mesh is
    watertight (alpha-wrapped) with consistent outward normals.
    """
    import trimesh as _tm

    if total <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    n_near = int(round(_SAMPLE_FRAC["near"] * total))
    n_hard = int(round(_SAMPLE_FRAC["hard"] * total))
    n_inside = int(round(_SAMPLE_FRAC["inside"] * total))
    n_outside = max(total - n_near - n_hard - n_inside, 0)
    n_surf = n_near + n_hard + n_inside

    # trimesh.sample uses the global numpy RNG; seed it for reproducibility.
    np_state = np.random.get_state()
    try:
        np.random.seed(seed & 0xFFFFFFFF)
        surf, fidx = _tm.sample.sample_surface(mesh, int(max(n_surf, 1)))
    finally:
        np.random.set_state(np_state)
    surf = np.asarray(surf, dtype=np.float64)
    fnorm = np.asarray(mesh.face_normals, dtype=np.float64)[np.asarray(fidx)]

    s_near, n0 = surf[:n_near], fnorm[:n_near]
    s_hard, n1 = surf[n_near:n_near + n_hard], fnorm[n_near:n_near + n_hard]
    s_in, n2 = surf[n_near + n_hard:n_surf], fnorm[n_near + n_hard:n_surf]

    near = s_near + n0 * rng.normal(0.0, sigma, size=(len(s_near), 1))
    hard = s_hard + n1 * rng.normal(0.0, sigma * 0.25, size=(len(s_hard), 1))
    inside = s_in - n2 * np.abs(rng.normal(0.0, sigma * 1.5, size=(len(s_in), 1)))
    # outside spans the padded range so boundary-touching tips get exterior context
    outside = rng.uniform(-_SAMPLE_MARGIN, 1.0 + _SAMPLE_MARGIN, size=(n_outside, 3))

    pts = np.concatenate([near, hard, inside, outside], axis=0)
    return np.clip(pts, -_SAMPLE_MARGIN, 1.0 + _SAMPLE_MARGIN)


def mesh_to_signed_distance(
    mesh_path: Path,
    resolution: int = 64,
    truncation: float = 4.0,
    *,
    thicken_fins: bool = False,
    fin_thickness: float = 0.015,
    thin_threshold: float = 0.01,
) -> np.ndarray:
    """True signed-distance TSDF queried directly against the mesh surface.

    Loads the mesh, normalizes it to the unit cube [0,1]^3 (same convention as
    ``mesh_to_occupancy``), then for each voxel-center query point computes the
    signed distance to the mesh surface. Result is converted to voxel units
    (multiply by ``resolution``) and clipped to ``[-truncation, truncation]``,
    then normalized to ``[-1, 1]``.

    Sign convention: **negative inside, positive outside** (so the surface is
    at iso=0 for marching cubes, matching ``voxel_to_tsdf``). libigl's default
    sign for ``SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER`` already matches this
    (negative inside, positive outside), so no flip is needed.

    Uses libigl's AABB-tree-based ``signed_distance`` with the fast winding
    number sign test. This is ~30-60x faster than trimesh's
    ``ProximityQuery.signed_distance`` and uses ~60x less memory (under 250 MB
    even for 76 k-face meshes at R=64). The previous trimesh path allocated
    O(n_query * n_faces) intermediates inside ``closest_point`` and OOM'd at
    R=64 on a 76 k-face mesh.
    """
    import igl

    mesh = _load_normalized_mesh_maybe_thickened(
        Path(mesh_path), thicken_fins, fin_thickness, thin_threshold
    )
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int64)

    R = int(resolution)
    coords = (np.arange(R, dtype=np.float64) + 0.5) / R
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)

    sd, _, _, _ = igl.signed_distance(
        points, V, F, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    )
    sd = np.asarray(sd, dtype=np.float32)  # negative inside / positive outside

    sdf_voxel = sd * R
    tsdf = np.clip(sdf_voxel / max(float(truncation), 1e-6), -1.0, 1.0)
    return tsdf.reshape(R, R, R).astype(np.float32)


def sample_query_points_and_sdf(
    mesh_path: Path,
    num_near_surface: int = 8192,
    num_uniform: int = 8192,
    near_surface_sigma: float = 0.03,
    seed: int = 0,
    *,
    thicken_fins: bool = False,
    fin_thickness: float = 0.015,
    thin_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a mixed pool of query points and their true mesh SDFs.

    Used by the implicit-decoder head: the model is supervised pointwise instead
    of on a dense voxel grid, so the supervision resolution decouples from the
    inference resolution. About half the points are sampled on the mesh surface
    and pushed off by Gaussian normal noise; the other half are uniformly drawn
    in the unit cube to give the decoder coverage of empty space.

    Returns
    -------
    points : float32 [num_near_surface + num_uniform, 3] in [0, 1]^3
    sdf    : float32 [...]  signed distance in **unit-cube** coordinates.
             Negative inside, positive outside (matches mesh_to_signed_distance).
    """
    import trimesh
    import igl

    mesh = _load_normalized_mesh_maybe_thickened(
        Path(mesh_path), thicken_fins, fin_thickness, thin_threshold
    )
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F_arr = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    rng = np.random.default_rng(seed)

    total_pool = int(num_near_surface) + int(num_uniform)
    points = _sample_query_points(mesh, total_pool, float(near_surface_sigma), rng, seed)

    sd, _, _, _ = igl.signed_distance(
        points, V, F_arr, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    )
    sd = np.asarray(sd, dtype=np.float32)
    return points.astype(np.float32), sd


def compute_implicit_cache_v2(
    mesh_path: Path,
    *,
    num_near_surface: int,
    num_uniform: int,
    near_surface_sigma: float,
    grid_resolution: int = 64,
    truncation: float = 4.0,
    seed: int = 0,
    thicken_fins: bool = False,
    fin_thickness: float = 0.015,
    thin_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot precompute: load + (optionally) thicken mesh once, then compute
    both the per-point (points, sdf-in-unit-cube-units) pool *and* the dense
    truncated SDF grid that the DiT will consume as its x_t-side input.

    Returns
    -------
    points       : float32 [num_near+num_unif, 3]   in [0, 1]^3
    sdf_uc       : float32 [num_near+num_unif]      unit-cube SDF (negative inside)
    tsdf_grid_R  : float32 [R, R, R]                clipped TSDF in [-1, 1]
                                                    (sdf * R / truncation, then clipped)
    """
    import igl
    import trimesh

    mesh = _load_normalized_mesh_maybe_thickened(
        Path(mesh_path), thicken_fins, fin_thickness, thin_threshold
    )
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    rng = np.random.default_rng(seed)

    # per-point pool: 4-way near-surface-heavy mixture (see _sample_query_points).
    # The old (num_near_surface, num_uniform) split is reinterpreted as a single
    # total pool that the mixture redistributes.
    total_pool = int(num_near_surface) + int(num_uniform)
    points = _sample_query_points(mesh, total_pool, float(near_surface_sigma), rng, seed)
    sd_points, _, _, _ = igl.signed_distance(
        points, V, F, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    )

    # dense grid TSDF (clipped)
    R = int(grid_resolution)
    coords = (np.arange(R, dtype=np.float64) + 0.5) / R
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    grid_points = grid.reshape(-1, 3)
    sd_grid, _, _, _ = igl.signed_distance(
        grid_points, V, F, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    )
    tsdf_grid = np.clip(
        (sd_grid.astype(np.float32) * R) / max(float(truncation), 1e-6), -1.0, 1.0
    ).reshape(R, R, R)

    return (
        points.astype(np.float32),
        np.asarray(sd_points, dtype=np.float32),
        tsdf_grid.astype(np.float32),
    )


def downsample_voxel(voxel: np.ndarray, target_resolution: int) -> np.ndarray:
    """Average-pool a boolean/float voxel grid down to target_resolution^3.

    The result is a soft occupancy in [0, 1] (fraction of source cells that
    were occupied).
    """
    arr = np.asarray(voxel, dtype=np.float32)
    R = arr.shape[0]
    if arr.shape != (R, R, R):
        raise ValueError(f"voxel must be cubic, got {arr.shape}")
    if R == target_resolution:
        return arr
    if R % target_resolution != 0:
        raise ValueError(f"source resolution {R} not divisible by target {target_resolution}")
    factor = R // target_resolution
    return arr.reshape(target_resolution, factor, target_resolution, factor, target_resolution, factor).mean(
        axis=(1, 3, 5)
    )


class PatchTokenizer:
    """Patchify / unpatchify a [B, C, R, R, R] volume into [B, N, C*P^3] tokens.

    Default config: resolution=64, patch=4 -> 16^3 = 4096 spatial tokens,
    each carrying P^3=64 channels per channel.

    The transformation is a pure spatial reshape (no learning), so the round
    trip is exact.
    """

    def __init__(self, resolution: int = 64, patch: int = 4) -> None:
        if resolution % patch != 0:
            raise ValueError(f"resolution {resolution} must be divisible by patch {patch}")
        self.resolution = resolution
        self.patch = patch
        self.grid = resolution // patch
        self.num_tokens = self.grid ** 3
        self.patch_volume = patch ** 3

    def patchify(self, volume: torch.Tensor) -> torch.Tensor:
        """[B, C, R, R, R] -> [B, N, C * P^3]"""
        if volume.ndim != 5:
            raise ValueError(f"expected 5D [B,C,R,R,R], got {tuple(volume.shape)}")
        B, C, D, H, W = volume.shape
        R, P = self.resolution, self.patch
        if (D, H, W) != (R, R, R):
            raise ValueError(f"spatial dims {(D,H,W)} != ({R},{R},{R})")
        # split each axis into (grid, patch)
        x = volume.reshape(B, C, R // P, P, R // P, P, R // P, P)
        # bring grid axes together (B, grid_d, grid_h, grid_w, C, P, P, P)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, self.num_tokens, C * self.patch_volume)
        return x

    def unpatchify(self, tokens: torch.Tensor, channels: int = 1) -> torch.Tensor:
        """[B, N, C * P^3] -> [B, C, R, R, R]"""
        if tokens.ndim != 3:
            raise ValueError(f"expected 3D [B,N,D], got {tuple(tokens.shape)}")
        B, N, D = tokens.shape
        if N != self.num_tokens:
            raise ValueError(f"token count {N} != expected {self.num_tokens}")
        if D != channels * self.patch_volume:
            raise ValueError(
                f"token dim {D} != channels({channels})*P^3({self.patch_volume})={channels * self.patch_volume}"
            )
        R, P = self.resolution, self.patch
        G = self.grid
        x = tokens.reshape(B, G, G, G, channels, P, P, P)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(B, channels, R, R, R)
