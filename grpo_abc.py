

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from trl import GRPOTrainer, GRPOConfig, TrlParser
from cadevolve_dataset import STLImageToCode, _get_patch_size, render_7view_montage, TOTAL_W, TOTAL_H, IMAGE_SIZE, AdaptiveScaler, _assert_image_geometry, _pad_to_multiple, _trimesh_to_o3d, transform_gt_mesh_cad_recodev2
from datasets import Dataset, Features, Value, Image as HFImage
from pathlib import Path

from qwen_vl_utils import smart_resize

from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
from dataclasses import dataclass
from transformers import HfArgumentParser

import torch
from torch import optim

from grpo_video_trainer import VideoTopSampleGRPOTrainer

import sys
sys.path.append("/home/jovyan/users/zhemchuzhnikov/miniconda3/envs/zhemchuzhnikov/lib/python3.10/site-packages")
import open3d as o3d
import trimesh
from utils_async import init_pool, close_pool, get_metrics_from_texts

import math
import numpy as np




CKPT = "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/work_dirs/qwen2vl_image2code_stls_v2_aug_updated/checkpoint-31063"
CKPT = "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/work_dirs/qwen2vl_image2code_stls_v2_aug/checkpoint-108000"
MODEL_ID= "Qwen/Qwen2-VL-2B-Instruct"
SEED = 99
VISION_PATCH_MULTIPLE = 14
ABC_ROOT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/ABCdataset/archives/abc_meshes/")


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
    gen_sample_steps: int =  25 // 3
    pool_size: int = 16

@dataclass
class ModelConfig:
    sft_path: str = "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/work_dirs/qwen2vl_image2code_stls_v2_aug_updated/final_model"

parser = TrlParser((GRPOConfig, RewardArgs, ModelConfig))
grpo, rargs, margs = parser.parse_args_and_config()
grpo.output_dir = "models/" + grpo.output_dir

#init_pool(rargs.pool_size)

sft_path = margs.sft_path

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
processor = AutoProcessor.from_pretrained(sft_path, padding_side="left", use_fast=True)

processor.image_processor.do_resize = False
processor.image_processor.do_center_crop = False
processor.image_processor.do_rescale = True
processor.image_processor.do_normalize = True

model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=sft_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True, local_files_only=True
)
model.enable_input_require_grads() 




VISION_PATCH_MULTIPLE = _get_patch_size(model)
print(f"[init] Vision patch multiple set to {VISION_PATCH_MULTIPLE}")

"""
#abc_500_samples = STLImageToCode(ABC_ROOT, img_size=IMAGE_SIZE, size=500, split_file=None)
abc_1_samples = STLImageToCode(ABC_ROOT, img_size=IMAGE_SIZE, size=1000, start=500001, split_file=None)
#train_deepcad = STLImageToCode(STLS_DP_ROOT, split_file=None, pickle_file='/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train/train_small.pkl', img_size=IMAGE_SIZE)
#train_combined = ConcatDataset([val_ds_cad_recode, train_deepcad])
#train_data = train_combined
train_data = abc_1_samples
INSTR = "Generate CadQuery v2 code for this 3-D shape. Return only Python code that assigns the final solid to variable `result`."
mesh_paths = [str(p) for p in train_data.items]


base = Dataset.from_list(
    [{"mesh_path": mp, "instruction": INSTR} for mp in mesh_paths],
    features=Features({
        "mesh_path": Value("string"),
        "instruction": Value("string"),
    })
)
"""
#### instead of process vision info of qwen vl-------
IMAGE_FACTOR = 28
MIN_PIXELS   = 4 * 28 * 28
MAX_PIXELS   = 16384 * 28 * 28

def qwen_align(img):
    h, w = img.height, img.width
    h2, w2 = smart_resize(h, w, factor=IMAGE_FACTOR,
                          min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if (h2, w2) != (h, w):
        img = img.resize((w2, h2))  # BICUBIC default in PIL
    return img

#### --------------------------------------------------

def render_one(ex):  
    stl_path = (ex["mesh_path"])
    try:
        """
        m = trimesh.load_mesh(stl_path, process=False)
        if m.vertices.size == 0 or max(m.extents) == 0:
            return {"image": None}
        m = transform_gt_mesh_cad_recodev2(m)
        o3 = _trimesh_to_o3d(m)
        #img = render_7view_montage(o3, thumb_size=THUMB_SIZE, pad=PAD, patch_mult=patch_mult)
        if o3.is_empty():
            return {"image": None}"""
        o3 = o3d.io.read_triangle_mesh(str(stl_path))
        img = render_7view_montage(o3, AdaptiveScaler(60, 200), thumb_size=IMAGE_SIZE)
        img = _pad_to_multiple(img, VISION_PATCH_MULTIPLE, bg="white") 
        img = qwen_align(img)
        _assert_image_geometry([img], [stl_path], multiple=VISION_PATCH_MULTIPLE)
        return {"image":img}
    except Exception as e:
        print(f"Render failed for {stl_path}: {e}")
        return {"image": None}
"""
features = Features({
    "mesh_path": Value("string"),
    "instruction": Value("string"),
    "image": HFImage(),           # IMPORTANT: image stays a real image column
})
ds = base.map(
    render_one,
    num_proc=156,
    features=features,
    writer_batch_size=64,
    #keep_in_memory=True
).filter(lambda ex: ex["image"] is not None)


ds_processed_path = "./rendered_abc_val_10K"
ds.save_to_disk(ds_processed_path)

"""
from datasets import load_from_disk
ds = load_from_disk("./rendered_abc_all")

def make_conversation(ex):
    conversation = [
        {"role": "user",
         "content": [
             {"type": "image"},
             {"type": "text", "text": ex["instruction"]}
         ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {"prompt": prompt}


ds = ds.map(
    make_conversation,
    num_proc=156,
    remove_columns=[],)


print(ds[0])
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
    elif mode == "auc_cd_combined":
        r = np.clip(-1/6 * np.log10(cd), 0, 1) * 2 + auc * 8
    else:
        r = 10.0 * float(iou)
    return float(np.clip(r, -10.0, 10.0))


def get_reward_function(failure_reward, iou_coef=10, cd_coef=0, auc_coef=0, nc_params=None):
    def combined_reward(completions, mesh_path, trainer_state=None, **kwargs):
        # Get individual rewards
        rewards = []
        # excepts = []
        pred_metrics = get_metrics_from_texts(
            completions, mesh_path, nc_params)
        # print("MESHES", pred_meshes, flush=True)
        for m in pred_metrics:
            reward = 0
            iou = m["iou"] if m is not None else None
            cd =  m["cd"] if m is not None else None
            auc =  m["auc"] if m is not None else None
            if auc is None or cd is None:
                reward = failure_reward
            else:
                reward = reward_from_metrics(cd, iou, auc=auc, mode="auc_cd_combined")
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
    print(f"\n[SAMPLE @ step {step}]\nMesh path : {mesh_path} \n{completion}\n", flush=True)

from grpo_loss import cppo_compute_loss, adv_select_top_samples
class TopSampleGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False,  **kwargs):
        return cppo_compute_loss(self, model, inputs, return_outputs=return_outputs)
    def select_top_samples(self, inputs, num_generations: int, top_samples: int=TOP_SAMPLES):
        return adv_select_top_samples(self, inputs, num_generations, top_samples)

    def _prepare_inputs(self, generation_batch):
        self.args.steps_per_generation = 1
        return super()._prepare_inputs(generation_batch)


from vllm_client_patch import patch_vllm_group_port, patch_vllm_for_videos
patch_vllm_for_videos()

# those parameters will be passed to vllm generation trainer
bad_words = ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in bad_words if processor.tokenizer.convert_tokens_to_ids(t) != processor.tokenizer.unk_token_id]

grpo.generation_kwargs = {
    "bad_words": bad_words,
}


trainer = VideoTopSampleGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
    args=grpo,
    optimizers=(optimizer, lr_scheduler),
    top_samples=TOP_SAMPLES,
    clip_cov=False
)

print(f"steps_per_generation value to ovverride: {trainer.args.steps_per_generation}")
trainer.train(
    resume_from_checkpoint="/workspace-SR008.nfs2/users/barannikov/trl_cadevolve/models/abc_all_gspo_temp1_top30_nc8cd2/checkpoint-5200"
)
