
import os, sys, random, json, warnings, time
from pathlib import Path
from functools import partial
from typing import List, Tuple, Dict, Any
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.transform import resize
import trimesh
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration,
    Trainer, TrainingArguments, TrainerCallback,
)

sys.path.append("/home/jovyan/users/zhemchuzhnikov/miniconda3/envs/zhemchuzhnikov/lib/python3.10/site-packages")
import open3d as o3d

CODE_ROOT   = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/maksim_sampling_rewritten_api")
STLS_ROOT   = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/stls_v2")

TRAIN_SPLIT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/CADEvolve/clustering/train_list")
VAL_SPLIT   = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/CADEvolve/clustering/val_list")

IMAGE_SIZE  = 130     # base thumb size (isometric is 2x)
PAD = 16
ROW1_W = 2*IMAGE_SIZE + PAD + 2 * IMAGE_SIZE  + 2 * PAD
ROW2_W = 4 * IMAGE_SIZE + 3 * PAD
TOTAL_W = max(ROW1_W, ROW2_W)   # = 568 for IMAGE_SIZE=130, PAD=16
TOTAL_H = 2*IMAGE_SIZE + PAD + IMAGE_SIZE  # = 406

TEXT_LABEL_HEIGHT = 18


def transform_gt_mesh_cad_recodev2(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(0.875 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3 = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces))
    o3.paint_uniform_color([0.6, 0.6, 0.6])
    o3.compute_vertex_normals()
    return o3

def _get_patch_size(model) -> int:
    """Read patch size from model.config.vision_config; fallback=14."""
    for k in ("patch_size", "patch_size_image", "patch_embed_size"):
        try:
            v = getattr(model.config.vision_config, k)
            if v is not None:
                return int(v)
        except Exception:
            pass
    return 14
    
def _pad_to_multiple(img: Image.Image, multiple: int, bg="white") -> Image.Image:
    """Pad (no resize) so width and height are multiples of `multiple`."""
    W, H = img.size
    W2 = ((W + multiple - 1) // multiple) * multiple
    H2 = ((H + multiple - 1) // multiple) * multiple
    if (W2, H2) == (W, H):
        return img
    canvas = Image.new("RGB", (W2, H2), bg)
    canvas.paste(img, (0, 0))
    return canvas

def _assert_image_geometry(images, names, multiple: int):
    """Ensure all PIL images have identical size and are aligned to `multiple`."""
    sizes = [(im.width, im.height) for im in images]
    uniq = sorted(set(sizes))
    bad_align = [(w, h) for (w, h) in sizes if (w % multiple) or (h % multiple)]
    if len(uniq) > 1 or bad_align:
        msg = [
            "[collate] Image geometry mismatch detected!",
            f"  unique sizes: {uniq}",
            f"  required multiple: {multiple}",
        ]
        try:
            paired = list(zip(names, sizes))
            msg.append("  per-sample sizes: " + ", ".join(f"{n}={s}" for n, s in paired))
        except Exception:
            pass
        if bad_align:
            msg.append(f"  not aligned to multiple: {bad_align}")
        raise RuntimeError("\n".join(msg))

# ---------------------------------------------------------------------
# Adaptive scaler (keep largest ≈100%, smallest ≈75% of canvas)
# ---------------------------------------------------------------------
class AdaptiveScaler:
    def __init__(self, e_min: float, e_max: float, eps: float = 1e-9):
        self.e_min = max(float(e_min), eps)
        self.e_max = max(float(e_max), self.e_min + eps)
        self.eps   = float(eps)

    @staticmethod
    def extent_of_o3d_mesh(mesh: o3d.geometry.TriangleMesh) -> float:
        aabb   = mesh.get_axis_aligned_bounding_box()
        extent = float(np.max(aabb.get_extent()))
        return extent

    def _target_visual_scale(self, E: float) -> float:
        if self.e_max - self.e_min < self.eps:
            return 1.0
        t = (E - self.e_min) / max(self.e_max - self.e_min, self.eps)
        return float(np.clip(0.75 + 0.25 * t, 0.75, 1.0))

    def scale_factor(self, E: float) -> float:
        vE = self._target_visual_scale(E)
        return (self.e_max / 200.0) * (vE / max(E, self.eps))

    def normalize_inplace(self, mesh: o3d.geometry.TriangleMesh) -> None:
        E = self.extent_of_o3d_mesh(mesh)
        s = self.scale_factor(E)
        mesh.scale(s, center=(0, 0, 0))
        mesh.translate([0.5, 0.5, 0.5])

# ---------------------------------------------------------------------
# 7-view renderer (legacy Visualizer)
# ---------------------------------------------------------------------

def _bake_vertex_shading(
    mesh: o3d.geometry.TriangleMesh,
    light_dir_world: np.ndarray,
    base_rgb=(0.6, 0.6, 0.6),
    ambient: float = 0.25
) -> o3d.geometry.TriangleMesh:
    """
    Writes per-vertex colors using simple Lambertian shading with a fixed world-space light.
    """
    m = o3d.geometry.TriangleMesh(mesh)            # copy
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    L = np.asarray(light_dir_world, float)
    L /= (np.linalg.norm(L) + 1e-12)
    N = np.asarray(m.vertex_normals)               # world-space normals
    ndotl = np.clip(N @ L, 0.0, 1.0)               # Lambert
    intensity = ambient + (1.0 - ambient) * ndotl
    base = np.array(base_rgb, float)[None, :]
    colors = np.clip(base * intensity[:, None], 0.0, 1.0)
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    return m


def _camera_extrinsic(front: np.ndarray):
    front = np.asarray(front, float)
    front /= (np.linalg.norm(front) + 1e-12)
    aux = np.array([0., 1., 0.]) if (abs(front[0]) < 0.1 and abs(front[1]) < 0.9) else np.array([0., 0., 1.])
    right = np.cross(aux, front); right /= (np.linalg.norm(right) + 1e-12)
    up = np.cross(front, right)
    R = np.column_stack((right, up, front)).T
    eye = np.array([0.5, 0.5, 0.5]) - 1.5 * front
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3]  = -R @ eye
    return extrinsic


def _render_view_mesh(mesh: o3d.geometry.TriangleMesh, front, img_res: int,
                      alpha: float = 0.6, flat_shading=True, bake_diag=False, light_world = np.array([1.0, 1.0, 1], dtype=float)):
    # If we bake lighting, turn off scene lights and render vertex colors only.
    if bake_diag:
        # light_world = _world_diag_light_from_front(front, toward=1.0)
        # light_world = np.array([1.0, -1.0, 1], dtype=float)
        # light_world = np.array([-1.0, -1.0, 1], dtype=float)
        # light_world = np.array([1.0, 1.0, 1], dtype=float)
        light_world /= (np.linalg.norm(light_world) + 1e-12)
        mesh_to_draw = _bake_vertex_shading(mesh, light_world, base_rgb=(0.6, 0.6, 0.6), ambient=0.25)
        flat_shading = True
    else:
        mesh_to_draw = mesh

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=max(500, img_res), height=max(500, img_res), visible=False)
    vis.add_geometry(mesh_to_draw)

    opt = vis.get_render_option()
    opt.background_color = np.ones(3)
    if flat_shading:
        opt.light_on = False          # show vertex colors as-is
    else:
        try:
            opt.light_on = True       # Phong
        except Exception:
            pass
    try:
        from open3d.visualization import MeshShadeOption
        opt.mesh_shade_option = MeshShadeOption.Phong
    except Exception:
        pass

    ctrl = vis.get_view_control()
    cam = ctrl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = _camera_extrinsic(np.asarray(front, float))
    try:
        ctrl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    except TypeError:
        ctrl.convert_from_pinhole_camera_parameters(cam)

    vis.poll_events(); vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = resize(img, (img_res, img_res), order=2, anti_aliasing=True, preserve_range=True).astype(np.float32)
    img = alpha * img + (1.0 - alpha) * 1.0
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def render_7view_montage(mesh_o3d: o3d.geometry.TriangleMesh,
                         scaler: AdaptiveScaler,
                         thumb_size: int = 130,
                         pad: int = 16,
                         bg_color: str = "white",
                         fg_color: str = "black",
                         alpha: float = 0.6) -> Image.Image:
    mesh = o3d.geometry.TriangleMesh(mesh_o3d)  # shallow copy
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.6, 0.6, 0.6])
    scaler.normalize_inplace(mesh)  # adaptive scale + translate

    views = {
        "Isometric": (1, 1, 1),
        "+Y": (0, 1, 0),
        "-Y": (0, -1, 0),
        "-Z": (0, 0, -1),
        "-X": (-1, 0, 0),
        "+Z": (0, 0, 1),
        "+X": (1, 0, 0),
    }
    tiles = {}
    for name, front in views.items():
        res = thumb_size * 2 if name == "Isometric" else thumb_size
        use_baked = (name != "Isometric")     # diagonal light only for orthographic views
        if name in ["+X", "+Y", "+Z"]:
            light_world = np.array([-1.0, -1.0, -1], dtype=float)
        else:
            light_world = np.array([1.0, 1.0, 1], dtype=float)
        tiles[name] = _render_view_mesh(mesh, front, res, alpha=alpha,
                                    flat_shading=False, bake_diag=use_baked, light_world = light_world)

    # Labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    text_h = font.getbbox("Hg")[3] + 4

    labeled = {}
    for name, img in tiles.items():
        canvas = Image.new("RGB", (img.width, img.height + text_h), bg_color)
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 0), f"View: {name}", font=font, fill=fg_color)
        canvas.paste(img, (0, text_h))
        labeled[name] = canvas

    # Layout: [Isometric | +Y | -Y] on row 1, then [-Z | -X | +Z | +X]
    iso_w, iso_h = labeled["Isometric"].size
    small_w, small_h = labeled["+Y"].size
    row1_w = iso_w + pad + 2 * small_w + 2 * pad
    row2_w = 4 * small_w + 3 * pad
    total_w = max(row1_w, row2_w)
    total_h = iso_h + pad + small_h

    combined = Image.new("RGB", (total_w, total_h), bg_color)
    x = 0
    combined.paste(labeled["Isometric"], (x, 0)); x += iso_w + pad
    combined.paste(labeled["+Y"], (x, 0));         x += small_w + pad
    combined.paste(labeled["-Y"], (x, 0))

    y = iso_h + pad; x = 0
    for key in ["-Z", "-X", "+Z", "+X"]:
        combined.paste(labeled[key], (x, y))
        x += small_w + pad

    # Border then **pad to vision patch multiple** (non-square, minimal pad)
    # combined = ImageOps.expand(combined, border=3, fill="black")
    # combined = _pad_to_multiple(combined, multiple=VISION_PATCH_MULTIPLE, bg="white")
    return combined



# ---------------------------------------------------------------------
# Dataset: pairs (montage image, target code)
# ---------------------------------------------------------------------
class STLImageToCode(Dataset):
    """
    Scans split file (comma-separated class folders). For each class folder F:
      stl:  STLS_ROOT/F/<stem>.stl
    """
    def __init__(self, stls_root: Path, split_file: Path = None,
                 img_size: int = IMAGE_SIZE, pickle_file=None, size=None):
        super().__init__()
        self.stls_root = stls_root
        self.img_size  = img_size
        self.scaler    = AdaptiveScaler(60, 200)

        self.items: List[dict] = []
        i=0
        go = True
        if split_file:
            folders = Path(split_file).read_text().strip().split(",")
            folders = [f.strip() for f in folders if f.strip()]
            for folder in folders:
                if go==False:
                    break
                sdir = stls_root / folder
                if not sdir.exists():
                    continue
                for stl in sorted(sdir.glob("*.stl")):
                    if stl.exists() and stl.stat().st_size > 0:
                        self.items.append(stl.resolve())
                        i+=1
                        if size and i >= size :
                            go = False
                            break
        elif pickle_file:
            with open(pickle_file, "rb") as f:
                annotations = pickle.load(f)

            self.items = [
                (stls_root / ann["mesh_path"]).resolve()
                for ann in annotations
                if (stls_root / ann["mesh_path"]).exists()
                and (stls_root / ann["mesh_path"]).stat().st_size > 0]   
        else:
            # take directly stls from stls_root
            for stl in stls_root.rglob("*.stl"):
                if stl.exists() and stl.stat().st_size > 0:
                    self.items.append(stl.resolve())
                    i += 1
                    if size and i >= size:
                        break


        if not self.items:
            raise RuntimeError(f"No stl files found using split {split_file}")

        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        stl_path = self.items[idx]

        try:
            m = trimesh.load_mesh(stl_path, process=False)
            if m.vertices.size == 0 or max(m.extents) == 0:
                return None
            m = transform_gt_mesh_cad_recodev2(m)
            o3 = _trimesh_to_o3d(m)
            #img = render_7view_montage(o3, thumb_size=THUMB_SIZE, pad=PAD, patch_mult=patch_mult)
            if o3.is_empty():
                return None
            img = render_7view_montage(o3, self.scaler, thumb_size=self.img_size)
            assert img.width  == TOTAL_W and img.height  == TOTAL_H
        except Exception as e:
            print(f"Render failed for {stl_path}: {e}")
            return None

        return {
            "image": img,                 # PIL.Image
            "instruction": "Generate CadQuery v2 code for this 3-D shape. Return only Python code that assigns the final solid to variable `result`.",            # target code
            "mesh_path": stl_path,
        }
