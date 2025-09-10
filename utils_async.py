import os
import time

os.environ["PYGLET_HEADLESS"] = "True"
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError, Process
from multiprocessing import get_context

import subprocess, sys
import base64, pickle, json, signal, select

import numpy as np
import faulthandler

faulthandler.enable()

"""
# for rendering only
from dataset_utils import render_mesh
import trimesh
from PIL import Image, ImageOps
import skimage
import open3d

def save_mesh_render(mesh, filename=None,
                     camera_distance=-1.8, front=[1, 1, 1],
                     width=500, height=500, img_size=128):
    os.makedirs("renders", exist_ok=True)
    if filename is None:
        timestamp = time.time()
        filename = f"render_{timestamp}.png"
    path = os.path.join("renders", filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
    mesh.compute_vertex_normals()
    img = render_mesh(mesh, camera_distance, front, width, height, img_size)
    img.save(path)
    return path"""

class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc

# process initializer used in case of forkserver
def init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    
    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq
    #from normal_metrics import compute_normals_metrics

    # make them available to your metric code
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['cq'] = cq
    #globals()['compute_normals_metrics'] = compute_normals_metrics


def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area over the curve (AOC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    #tol = 0.01 * max(gt_mesh.extents.max(), pred_mesh.extents.max())  # 1% of the mesh extent
    tol = pred_mesh.extents.max() * tol  / 100

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)

    # normals of sampled points
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)
    # get the indices of the neighbors for each ground-truth point

    valid_pred_normals = []
    valid_gt_normals = []
    valid_gt_points = []
    valid_pred_points = []

    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs] # candidates

        valid_gt_normals.append(gn)
        dots = (pn_neighbors * gn).sum(axis=1)  # (k,)
        best_idx = np.argmax(dots)  # index of the best aligned normal

        valid_pred_normals.append(pn_neighbors[best_idx])  # (3,)

        valid_gt_points.append(gt_points[i])  # (3,)
        valid_pred_points.append(pred_points[idxs[best_idx]])  # (3,)

    if len(valid_pred_normals) == 0:
        return None, None, None

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    valid_gt_points = np.vstack(valid_gt_points)
    valid_pred_points = np.vstack(valid_pred_points)

    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100
    print(f"Number of points with no neighbors within tol: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")

    
    
    # compute cosine similarity
    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)
    
    # distribution of angles between normals
    angles = np.arccos(cos_sim)
    angles = np.sort(angles)

    # add invalid points to the end of the array with max angle (pi)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N+1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0],   cdf,   [1.0]))
    auc_normalized = trapz(y, x) / np.pi  # Normalize by the maximum possible aoc (which is pi)

    #we want to maximize the AUC
    #aoc_normalized = 1 - auc_normalized
    # plot the aoc
    #if aoc_normalized > 0.3:
        #print(f"HIGH aoc: {aoc_normalized:.2f}")
        #plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)


    return auc_normalized, mean_cos_sim, per_invalid


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd



def transform_real_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    mesh.apply_scale(2.0 / max(mesh.extents))  # Normalize to [-1, 1]
    return mesh


def transform_gt_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh

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

def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)  # Normalize to [0, 1]
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh

def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def code_to_mesh_and_brep_less_safe(code_str, var_name="result"):
    safe_ns = {"cq": cq}
    ns=safe_ns.copy()
    #print(f"Executing code {code_str}")
    try:
        exec(code_str, ns)
        mesh = compound_to_mesh(ns[var_name].val())
        # export files if needed
        # mesh.export(mesh_path)
        return mesh
    except Exception as e:
        print(f"Error executing CadQuery code : {e}")
        return None


def get_metrics_from_single_text(text, gt_file, n_points, nc_params=None, var_name="result"):

    gt_file = os.path.abspath(gt_file)
    base_file = os.path.basename(gt_file).rsplit('.stl', 1)[0]

    #print(f"computing metrics for file: {gt_file}", flush=True)
    
    #t_cad = time.perf_counter()
    try:
        # execute cadquery code
        pred_mesh = code_to_mesh_and_brep_less_safe(text, var_name)
    except Exception as e:
        return dict(file_name=base_file, cd=None, iou=None, auc=None)
    #print(f"[TIME] cad_exec: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")

    if pred_mesh is None:
        #print("Skipping metrics: invalid prediction", flush=True)
        return dict(file_name=base_file, cd=None, iou=None, auc=None)
    #t_met = time.perf_counter()
    cd, iou, auc = None, None, None
    try: 
        gt_mesh = trimesh.load_mesh(gt_file)

        #gt_mesh = transform_gt_mesh(gt_mesh)
        gt_mesh = transform_gt_mesh_cad_recodev2(gt_mesh)
        #print(f"normalized gt_meshes extents : {gt_mesh.extents}")
        #print("Loaded and normalized ground truth", flush=True)
        
        pred_mesh = transform_pred_mesh(pred_mesh)
        #print(f"normalized pred_mesh extents : {pred_mesh.extents}")
        #print("Normalizing prediction", flush=True) 

        #save_mesh_render(pred_mesh)

        cd = compute_cd(gt_mesh, pred_mesh, n_points)
        try:
            iou = compute_iou(gt_mesh, pred_mesh)
        except Exception as e:
            print(f"IoU error for {base_file}: {e}", flush=True)
            iou = None
        if nc_params and nc_params["get_nc" ] == True:
            auc, _, _ = compute_normals_metrics(gt_mesh, pred_mesh, n_points=nc_params.get("n_points", n_points), tol=nc_params.get("tol", 5))

    except Exception as e:
        print(f"error for {base_file}: {e}", flush=True)
        pass

    #print(f"[TIME] metric computation without cadquery: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")
    finally:
        try:
            if gt_mesh is not None:
                del gt_mesh
            if pred_mesh is not None:
                del pred_mesh
        except:
            pass
    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc)




POOL = None

def init_pool(max_workers):
    print("Initializing POOL", flush=True)
    ctx = get_context("forkserver")
    global POOL
    if POOL is None:
        #ctx = get_context("spawn")
        POOL = NonDaemonPool(
            processes=max_workers,
            initializer=init_worker,
            context=ctx,
        )
        print("POOL Initialized", flush=True)
    return POOL

def close_pool():
    global POOL
    if POOL is not None:
        POOL.close()
        POOL.join()
        POOL = None

"""
_child = None

def _ensure_child(env):
    global _child
    if _child is None or _child.poll() is not None:
        print(f"launching a fresh python interpreter with the metrics server")
        _child = subprocess.Popen(
            [sys.executable, "-u", "-m", "metrics_server", "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env=env,
        )
    return _child


def _read_stderr(p, max_bytes=65536, timeout=0.2):
    import select
    buf = bytearray()
    # wait until some data is available, up to timeout
    r, _, _ = select.select([p.stderr], [], [], timeout)
    while r and max_bytes > 0:
        chunk = p.stderr.read1(min(4096, max_bytes))  # non-blocking-ish
        if not chunk:
            break
        buf += chunk
        max_bytes -= len(chunk)
        # drain whatever else is immediately available
        r, _, _ = select.select([p.stderr], [], [], 0)
    return buf.decode(errors="replace")

def timed_subprocess_text(arg, timeout=60):
    global _child
    env = os.environ.copy()
    #env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(sys.path))
    p = _ensure_child(env)

    # send one line
    payload = base64.b64encode(pickle.dumps(arg)) + b"\n"
    p.stdin.write(payload); p.stdin.flush()

    # wait with hard timeout
    r, _, _ = select.select([p.stdout], [], [], timeout)
    if not r:
        if p.poll() is not None: 
            err = _read_stderr(p)
            if err: print(err, flush=True)
            try: p.wait(timeout=0.1)
            finally: _child = None
            return "__CRASH__"
        os.killpg(p.pid, signal.SIGKILL)   # kill group
        _child = None                      # force fresh child next call
        return "__TIMEOUT__"

    line = p.stdout.readline()
    err_output = _read_stderr(p, 65536, 0.2)
    #if err_output: print(f"[child-stderr after exit]\n{err_output}", flush=True)
    if line == b"": # EOF
        _child = None 
        return "__CRASH__"

    try:
        r = json.loads(line)
        return r
    except Exception as e:
        print(f"exception in reward server {e}")
        return "__CRASH__"
"""

def timed_process_text(arg, timeout=60):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result  


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)          # may be None or a string
    finally:
        conn.close()


def get_metrics_from_texts(texts, meshes, nc_params=None, max_workers=None, var_name="result"):
    print(f"[POOL] POOL size={POOL._processes} pid={os.getpid()}")
    #t0 = time.perf_counter()
    #print(f"Example of one generated code : {texts[0]} for file : {meshes[0]}")
    n_points = 16384
    args = [
        (text, gt, n_points, nc_params, var_name)
        for text, gt in zip(texts, meshes)
    ]
    async_results = [POOL.apply_async(timed_process_text, args=(arg,)) for arg in args]
    results = []
    for res in async_results:
        output = res.get()
        if output == "__TIMEOUT__" or output == "__CRASH__":
            print(f"[{output}] metrics task computation ERROR, skipping", flush=True)
            results.append(dict(file_name=None, cd=None, iou=None, auc=None))
        else:
            results.append(output)

    #wait = time.perf_counter() - t0 
    #print(f"TIME to get metrics for {len(texts)} samples : {wait}")

    return results