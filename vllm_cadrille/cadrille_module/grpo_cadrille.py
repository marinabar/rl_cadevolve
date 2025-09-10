import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from trl import GRPOConfig, TrlParser
sys.path.append("/workspace-SR008.nfs2/users/barannikov/trl_cadevolve")

from grpo_video_trainer import VideoTopSampleGRPOTrainer

from cadevolve_dataset import STLImageToCode, _get_patch_size, render_7view_montage, TOTAL_W, TOTAL_H, IMAGE_SIZE, AdaptiveScaler, _assert_image_geometry, _pad_to_multiple, _trimesh_to_o3d, transform_gt_mesh_cad_recodev2
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
from pathlib import Path

from qwen_vl_utils import smart_resize

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, GenerationConfig
from dataclasses import dataclass
from transformers import HfArgumentParser

import torch
from torch import optim


sys.path.append("/home/jovyan/users/zhemchuzhnikov/miniconda3/envs/zhemchuzhnikov/lib/python3.10/site-packages")
import open3d as o3d
import trimesh
from utils_async import init_pool, close_pool, get_metrics_from_texts, transform_real_mesh
from cadrille_dataset import RealDatasetMM

import math
import numpy as np




MODEL_PATH = "/workspace-SR008.nfs2/users/barannikov/cadrille/models/cadrille"
SEED = 99
VISION_PATCH_MULTIPLE = 14
STLS_ROOT   = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/stls_v2")
STLS_DP_ROOT = Path("/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train")
TRAIN_SPLIT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/CADEvolve/clustering/train_list")
VAL_SPLIT   = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/CADEvolve/clustering/val_list")

SEED = 96
TOP_SAMPLES = 4



@dataclass
class RewardArgs:
    failure_reward: float = -10.0
    iou_coef: float = 10.0
    cd_coef: float = 0.0
    auc_coef: float = 0.0
    get_nc: bool = False
    # how many points to sample from surface
    nc_n_points: int = 16384
    # what percentage of overall mesh extents to look for neighbors in
    nc_tol: int = 5
    gen_sample_steps: int = 25 // 3
    pool_size: int = 16

@dataclass
class ModelConfig:
    sft_path: str = "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/work_dirs/qwen2vl_image2code_stls_v2_aug_updated/final_model"
    group_port: int = 51216

@dataclass
class TrainingArgs:
    adv_multiplier: int = 1
    clip_cov: bool = False


parser = TrlParser((GRPOConfig, RewardArgs, ModelConfig, TrainingArgs))
grpo, rargs, margs, targs = parser.parse_args_and_config()
grpo.output_dir = "models/" + grpo.output_dir

init_pool(rargs.pool_size)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


from cadrille_plugin.cadrille import Cadrille
sys.path.append("/workspace-SR008.nfs2/users/barannikov/trl_cadevolve/dev/cadrille")  # parent of cadrille_plugin
import cadrille_plugin


model = Cadrille.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",)
model.gradient_checkpointing_disable()
model.enable_input_require_grads() 
model.config.use_cache = True
model.config.gradient_checkpointing = False
model.freeze_pc()


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                            min_pixels=256 * 28 * 28,
                                            max_pixels=1280 * 28 * 28,
                                            padding_side="left",
                                            )
processor.tokenizer.padding_side = "left"

processor.image_processor.size.update({
    "shortest_edge": 3136,
    "longest_edge": 12845056,
})

print(processor.image_processor)

from patch_vllm import patch_vllm_proccessor
patch_vllm_proccessor(processor)


from vllm_client_patch import patch_vllm_group_port, patch_vllm_for_videos
patch_vllm_group_port(margs.group_port)
patch_vllm_for_videos()



VISION_PATCH_MULTIPLE = _get_patch_size(model)
print(f"[init] Vision patch multiple set to {VISION_PATCH_MULTIPLE}")

train_data = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name="train_small.pkl", n_points=256, mode="img", noise_scale_pc=0.01, size=100)

#val_ds_cad_recode = STLImageToCode(STLS_ROOT, split_file=VAL_SPLIT, img_size=IMAGE_SIZE, size=10)
#train_deepcad = STLImageToCode(STLS_DP_ROOT, split_file=None, pickle_file='/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train/train_small.pkl', img_size=IMAGE_SIZE)
#train_combined = ConcatDataset([val_ds_cad_recode, train_deepcad])
#train_data = train_combined
INSTR = "Generate cadquery code"
mesh_paths = [os.path.join(train_data.path, train_data.annotations[a]['mesh_path']) for a in train_data.annotations]


base = Dataset.from_list(
    [{"mesh_path": mp, "instruction": INSTR} for mp in mesh_paths],
    features=Features({
        "mesh_path": Value("string"),
        "instruction": Value("string"),
    })
)

#### instead of process vision info of qwen vl-------
IMAGE_FACTOR = 28
MIN_PIXELS   = 4 * 28 * 28
MAX_PIXELS   = 16384 * 28 * 28

def qwen_align(img):
    h, w = img.height, img.width
    h2, w2 = smart_resize(h, w, factor=IMAGE_FACTOR,
                          min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if (h2, w2) != (h, w):
        img = img.resize((w2, h2))
    return img

#### --------------------------------------------------

def render_one(ex):  
    stl_path = ex["mesh_path"]
    try:
        mesh = trimesh.load_mesh(stl_path)
        mesh = transform_real_mesh(mesh)
        out = train_data.get_img(mesh)
        video = out["video"]
        video = [qwen_align(frame) for frame in video]
        return {"video": video}
    except Exception as e:
        print(f"Render failed for {stl_path}: {e}")
        return {"video": None}
        
"""
features = Features({
    "mesh_path": Value("string"),
    "instruction": Value("string"),
    "video": Sequence(HFImage())
})

ds = base.map(
    render_one,
    num_proc=156,
    features=features,
    writer_batch_size=64,
    #keep_in_memory=True
).filter(lambda ex: ex["video"] is not None)


ds_processed_path = "./rendered_1view_deepcad_test"
ds.save_to_disk(ds_processed_path)
"""

from datasets import load_from_disk
ds = load_from_disk("/workspace-SR008.nfs2/users/barannikov/trl_cadevolve/rendered_1view_deepcad_test")

def make_conversation(ex):
    conversation = [
        {"role": "user",
         "content": [
             {"type": "video", 'fps': 1.0},
             {"type": "text", "text": ex["instruction"]}
         ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt}


ds = ds.map(
    make_conversation,
    num_proc=156,
    remove_columns=[],)


print(ds[0])

#dataset = ds.train_test_split(test_size=100, seed=42)
#train_dataset, eval_dataset = dataset["train"], dataset["test"]
from datasets import concatenate_datasets
k = 300
#train_dataset = concatenate_datasets([ds]*k)

train_dataset = ds

# -------------- coefficients for the geometric reward -------------
def reward_from_metrics(cd: float, iou: float, auc: float = 0, mode: str = "default") -> float:
    #if math.isnan(iou): iou = 0.0
    if cd is None or math.isnan(cd) or cd <= 0: cd = 1.0
    if mode == "10_iou":
        r = 10.0 * float(iou)
    elif mode == "cd_to_reward":
        ln = math.log(max(cd, 1e-8))
        denom = (ln - 1.0)
        if abs(denom) < 1e-4: denom = 1e-4 if denom >= 0 else -1e-4
        r = 10.0 * (1.0 + 1.0 / denom)
    elif mode == "iou":
        r = float(iou)
    elif mode == "10_normal_auc":
        r  = 10.0 * auc
    else:
        r = 10.0 * float(iou)
    return float(np.clip(r, -10.0, 10.0))

def update_step_tol(step):
    if step >= 150:
        tol = 2
    elif step >= 75:
        tol = 2
    else:
        tol = nc_params["tol"]
    #print(f"[tol] step {step}: tol={tol}")
    return tol


def reward_from_auc(auc, step, scale=10.0, eps=1e-8, tau=3e-3):
    auc = np.asarray(auc, dtype=float)
    z = np.clip((auc - 0.5) / 0.5, eps, 1 - eps)

    # scheduler: ~1 before 50 (minimal), mild after 50, strong after 100
    s = 1.0 \
        + 0.6 / (1.0 + np.exp(-(step - 50) / 10.0)) \
        + 1.6 / (1.0 + np.exp(-(step - 100) / 10.0))

    base = scale * (0.5 + 0.5 * z**s)                        # power sharpening
    # gate ensures rewards > 9.5 occur only when AUC > 0.975
    gate = 1.0 / (1.0 + np.exp(-(auc - 0.975) / tau))

    # ≤0.975: cap at 9.5 (don’t inflate mid-high AUCs)
    below = auc <= 0.975
    capped = np.where(below, np.minimum(base, 9.5), base)
    # >0.975: reveal the portion above 9.5 smoothly
    rewarded = np.where(below, capped, 9.5 + (base - 9.5) * gate)
    return rewarded


def get_reward_function(failure_reward, iou_coef=10, cd_coef=0, auc_coef=0, nc_params=None):
    def combined_reward(completions, mesh_path, trainer_state=None, **kwargs):
        # Get individual rewards
        rewards = []
        # excepts = []
        if nc_params.get("get_nc") == True:
            updt_tol = update_step_tol(step=getattr(trainer_state, "global_step", 0))
            nc_params["tol"] = updt_tol
        pred_metrics = get_metrics_from_texts(
            completions, mesh_path, nc_params, var_name="r")
        # print("MESHES", pred_meshes, flush=True)
        for m in pred_metrics:
            reward = 0
            iou = m["iou"] if m is not None else None
            cd =  m["cd"] if m is not None else None
            auc =  m["auc"] if m is not None else None
            if iou is None:
                reward = failure_reward
            else:
                reward = reward_from_metrics(cd, iou, auc=auc, mode="10_iou")
                #reward = reward_from_auc(auc=auc, step=trainer_state.global_step)
            rewards.append(float(reward))
        print(f"Rewards : {rewards}\n\n\n")

        # ---- print one sample every 50 steps ----
        top_idx = rewards.index(max(rewards))
        top_generation = completions[top_idx]
        top_mesh_path = mesh_path[top_idx]
        _maybe_print_sample(top_generation, top_mesh_path, step=trainer_state.global_step)

        return rewards
    return combined_reward

nc_params = {
    "get_nc": rargs.get_nc,
    "n_points" : rargs.nc_n_points, 
    "tol" : rargs.nc_tol,
}


reward_fn = get_reward_function(failure_reward=rargs.failure_reward, iou_coef=rargs.iou_coef, cd_coef=rargs.cd_coef, auc_coef=rargs.auc_coef, nc_params=nc_params)


optimizer = optim.AdamW(model.parameters(), lr=grpo.learning_rate)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
lr_scheduler = None


#----------- callback for generating samples
def _maybe_print_sample(completion, mesh_path, step, every=rargs.gen_sample_steps):
    if step == 0 or step % every != 0:
        return
    print(f"\n[SAMPLE @ step {step}]\n Mesh path : {mesh_path} \n {completion}\n", flush=True)

# those parameters will be passed to vllm generation trainer
bad_words = ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in bad_words if processor.tokenizer.convert_tokens_to_ids(t) != processor.tokenizer.unk_token_id]
'''
grpo.generation_kwargs = {
    "bad_words": bad_words,
    "suppress_tokens": ids,

}'''


trainer = VideoTopSampleGRPOTrainer(
    clip_cov=targs.clip_cov,
    top_samples=TOP_SAMPLES,
    model=model,
    processing_class=processor,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
    args=grpo,
    optimizers=(optimizer, lr_scheduler),
)

print(f"steps_per_generation value to ovverride: {trainer.args.steps_per_generation}")
trainer.train()
