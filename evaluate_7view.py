import os, numpy as np, torch
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from pathlib import Path

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


from cadevolve_dataset import STLImageToCode, IMAGE_SIZE
from utils_async import get_metrics_from_texts, init_pool

ABC_ROOT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/ABCdataset/archives/abc_meshes/")
CKPT_PATH = "/workspace-SR008.nfs2/users/barannikov/trl_cadevolve/models/abc_all_gspo_temp1_top30_nc8cd2_resume/checkpoint-8200"
BATCH_SIZE = 8
DEVICE = "cuda"

processor = AutoProcessor.from_pretrained(CKPT_PATH, trust_remote_code=True)

from datasets import load_from_disk
ds = load_from_disk("./rendered_abc_val_10K")

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


ds_eval = ds.map(
    make_conversation,
    num_proc=156,
    remove_columns=[],)


def collate_img_qwen(batch, processor):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    mesh_paths = [example["mesh_path"] for example in batch]
    prompts_text = [example["prompt"] for example in batch]

    images = [example.get("image") for example in batch]
    kwargs = {"images": [[img] for img in images]}

    prompt_inputs = processor(
        text=prompts_text,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        **kwargs,
    )
    prompt_inputs["mesh_path"] = mesh_paths
    return prompt_inputs


model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=CKPT_PATH,
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    local_files_only=True
).to("cuda")

eval_loader = DataLoader(
    ds_eval,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=20,
    pin_memory=True,
    collate_fn=partial(collate_img_qwen, processor=processor),
)

nc_params = {
    "get_nc": True,
    "n_points" : 8192, 
    "tol" : 5,
}

def evaluate_model_mm2_img(config, model, processor, eval_loader):
    model.eval()
    ious, cds, aucs = [], [], []
    aucs_mod = []
    n_incorrect, n_failed_intersect = 0, 0

    bad_words_ids = [[getattr(model.config, "vision_token_id")]] \
        if hasattr(model.config, "vision_token_id") else None

    with torch.inference_mode():
        for batch in tqdm(eval_loader):
            inputs = {
                k: v.to(model.device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            mesh_paths = batch["mesh_path"]
            print(f"")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                max_new_tokens=getattr(config, "max_new_tokens", 768),
                temperature=getattr(config, "temperature", 1.0),
                do_sample=True,
                top_p=getattr(config, "top_p", 0.99),
                top_k=getattr(config, "top_k", 30),
                bad_words_ids=bad_words_ids,
            )

            # trim prompts
            generated_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids
                in zip(inputs["input_ids"], generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            results = get_metrics_from_texts(py_strings, mesh_paths, nc_params)
            for res in results:
                if res is None or res.get("auc") is None:
                    n_incorrect += 1
                    continue
                iou = res["iou"]
                cd  = res["cd"]
                auc = res["auc"]
                if not iou or iou < 0:
                    n_failed_intersect += 1
                    cds.append(cd)
                    aucs.append(auc)
                else:
                    ious.append(iou)
                    cds.append(cd)
                    aucs.append(auc)

    # aggregates
    iou_mean, iou_med = (float(np.mean(ious)) if ious else float("nan"),
                         float(np.median(ious)) if ious else float("nan"))
    cd_mean, cd_med   = (float(np.mean(cds)) if cds else float("nan"),
                         float(np.median(cds)) if cds else float("nan"))
    auc_mean, auc_med   = (float(np.mean(aucs)) if aucs else float("nan"),
                         float(np.median(aucs)) if aucs else float("nan"))
    frac_intersect    = n_failed_intersect / 1000
    frac_incorrect    = n_incorrect / 1000

    print(f"IoU mean {iou_mean:.6f}, median {iou_med:.6f}")
    print(f"CD mean {cd_mean:.6f}, median {cd_med:.6f}")
    print(f"AUC of Normals mean {auc_mean:.6f}, median {auc_med:.6f}")
    print(f"Invalid generations fraction: {frac_incorrect:.6f}")
    print(f"Intersect failure fraction: {frac_intersect:.6f}")

    model.train()
    return ious, cds, frac_incorrect, frac_intersect


def run_eval():
    print("Starting evaluation")

    init_pool(25)
    experiment = None
    try:
        from comet_ml import start
        experiment = start(
            api_key="CfQGtyWGF13CZEsUvXBeuPaSf",
            project_name="cadevolve",
            workspace="marinabar",
        )
        experiment.set_name(EvalConfig.name)
        experiment.log_parameters(vars(EvalConfig))
    except Exception:
        pass

    ious, cds, frac_bad, frac_inter = evaluate_model_mm2_img(
        EvalConfig, model, processor, eval_loader
    )

    if experiment:
        experiment.log_metrics({
            "iou_mean": float(np.mean(ious)) if ious else float("nan"),
            "iou_median": float(np.median(ious)) if ious else float("nan"),
            "cd_mean": float(np.mean(cds)) if cds else float("nan"),
            "cd_median": float(np.median(cds)) if cds else float("nan"),
            "invalid_frac": frac_bad,
            "intersect_fail_frac": frac_inter,
            "num_eval": len(ds_eval),
        })

class EvalConfig:
    name = "eval-abc-500k-501k"
    temperature = 1.0
    do_sample = True
    top_p = 0.99
    top_k = 30
    max_new_tokens = 1400
    batch_size = BATCH_SIZE

if __name__ == "__main__":
    run_eval()