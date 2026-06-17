# VESFlow: Velocity Score-Based Guidance for Safe Mean-Flow / FLUX Generation

Reference implementation for the paper. Our method has two variants —
**VESFlow** (basic) and **VESFlow_str** (stronger) — and we compare against
**SGF**, **STG**, **SAFREE**, and **Semantic Surgery (SS)** on top of the
**MeanFlow** rectified-flow distillation of FLUX.1-lite-8B.

---

## 1 · Repository layout

```
VESFlow/
├── model/                        # FLUX transformer + pipeline (from T2I-Distill)
│   ├── mmdit_flux.py
│   ├── pipeline.py
│   ├── attn_processor.py
│   └── utils.py
│
├── sample_meanflow_safree.py     # main pipeline (VESFlow + SGF + STG + SAFREE + SS)
│
├── safe_denoiser.py              # VESFlow / VESFlowStr + scorer classes
├── sgf_stg_meanflow.py           # SGFMeanFlow + STGMeanFlow
├── embedding_modifier.py         # SAFREE + Semantic Surgery (prompt-embedding modifiers)
├── scorers.py                    # standalone Q16 / LAION-NSFW scorers
│
├── eval_sgf_stg_meanflow.py      # driver: SGF/STG/SAFREE/base
├── eval_meanflow_sgf_full.py     # full sweep driver (RAB + COCO + FID)
├── eval_metrics.py               # CLIPScore + FID
├── eval_q16_asr_tr.py            # Q16 (violence/inappropriate) evaluator
├── eval_mllm_safety.py           # LLaVA-1.5 cross-eval (yes/no)
│
├── checkpoints/
│   └── q16/prompts.p             # Q16 soft prompts (Schramowski et al. 2022)
│
├── datasets/                     # benchmark prompt CSVs (see § 4)
│   ├── nudity-ring-a-bell.csv
│   ├── nudity.csv
│   ├── mma-diffusion-nsfw-adv-prompts.csv
│   ├── i2p.csv
│   └── violence-ring-a-bell.csv
│
└── README.md
```

---

## 2 · Setup

```bash
# Python 3.10+
pip install torch torchvision
pip install diffusers transformers accelerate
pip install pandas pillow ftfy regex onnxruntime
pip install ultralytics                              # for STG NudeNet-YOLO
pip install git+https://github.com/openai/CLIP.git   # provides `clip` package
pip install cleanfid                                 # for FID
```

The two backbone weights are pulled automatically by `MeanFlowSAFREEPipeline`:

| HF repo | Used as |
|---|---|
| `Freepik/flux.1-lite-8B` | FLUX VAE + tokenizers + text encoders + base transformer |
| `Alibaba-DAMO-Academy/T2I-Distill` (`flux1lite_meanflow.pt`) | MeanFlow distilled weights |

Other Hugging Face models lazily loaded as needed:

| Model | Required for |
|---|---|
| `openai/clip-vit-large-patch14` | CLIPVelocityScorer / Q16 / LAION-NSFW |
| `llava-hf/llava-1.5-7b-hf` | `eval_mllm_safety.py` cross-eval |

---

## 3 · Checkpoints to download

Place these under `VESFlow/checkpoints/` (paths can be overridden via CLI).

### Required for Ours / VESFlow (nudity)

* **LAION-NSFW head** (`laion_nsfw_clipL.pt`, ~9 MB) — paired with CLIP-L/14:
  source <https://github.com/LAION-AI/CLIP-based-NSFW-Detector>
  Path used by default: `checkpoints/laion_nsfw/laion_nsfw_clipL.pt`

### Required for Ours / VESFlow (violence)

* **Graphic-harm head** (`laion_trained_graphicharm_clipL.pt`) — our trained
  head (pos = violence + shocking + self-harm, neg = sexual + hate +
  harassment + illegal), CLIP-L/14 features → 4-layer MLP → sigmoid.
  Release link: TODO.
  Pass to scorer via `--score_guide_laion_head_path`.

### Required for STG (in-loop NudeNet YOLO detector)

* **NudeNet-v3.4 640m YOLO** (`nudenet_640m.pt`) — used by STG-Nudity:
  source <https://github.com/notAI-tech/NudeNet>
  Path used by default: `checkpoints/nudenet/nudenet_640m.pt`

### Required for evaluation

* **SAFREE binary NudeNet classifier** (`nudenet_classifier_model.onnx`) —
  the binary nude/safe ONNX classifier used by SAFREE for evaluation:
  source <https://github.com/jaehong31/SAFREE>
  Pass via `--nudenet_path`.

### Required for SGF (paper-faithful reference images)

I2P "sexual" subset for SGF-nudity, I2P "violence" subset for SGF-violence:

```bash
# Inverse-Prompt I2P from https://huggingface.co/datasets/AIML-TUDA/i2p
# Filter per category:
python -c "
import pandas as pd
df = pd.read_csv('datasets/i2p.csv')
df[df['categories'].fillna('').str.contains('sexual', case=False)].to_csv('datasets/i2p_sexual.csv', index=False)
df[df['categories'].fillna('').str.contains('violence', case=False)].to_csv('datasets/i2p_violence.csv', index=False)
"
# Then generate reference images for each filter set with a base SD1.5/SDXL.
```

Already-prepared dirs are available on request.

### Optional (Q16-as-scorer experiments)

* `checkpoints/q16/prompts.p` (included — small file, 2 × 768)

---

## 4 · Benchmarks

| CSV | Concept | Eval | n |
|---|---|---|---|
| `nudity-ring-a-bell.csv` | Nudity (RAB-adv) | NudeNet binary classifier | 79 |
| `nudity.csv` | Nudity (orig RAB targets) | NudeNet | 79 |
| `mma-diffusion-nsfw-adv-prompts.csv` | Nudity (MMA stealth) | NudeNet | first 400 used |
| `i2p.csv` | I2P 10 categories | — | reference only |
| `violence-ring-a-bell.csv` | Violence (RAB-adv) | Q16 inappropriate | 250 |

For MS-COCO 10K FID/CLIP we use the same 10K subset CSV as in the paper
(`coco_30k_10k.csv`); see `eval_meanflow_sgf_full.py` for the path stub.

---

## 5 · Running the methods

All methods share `--seed 42`, `--steps 4`, `--guidance_scale 3.5` (MeanFlow defaults).

### 5.1 Baseline / SAFREE / SGF / STG (driver: `eval_sgf_stg_meanflow.py`)

```bash
# Baseline (no method)
python eval_sgf_stg_meanflow.py \
  --method base \
  --dataset_name ring-a-bell --csv datasets/nudity-ring-a-bell.csv \
  --steps 4 --guidance_scale 3.5 --seed 42 \
  --metrics nudenet clip \
  --nudenet_path /path/to/nudenet_classifier_model.onnx \
  --save_path results/base

# SGF (paper-faithful — pass these flags explicitly, defaults are weaker)
python eval_sgf_stg_meanflow.py \
  --method sgf \
  --unsafe_images datasets/i2p_sexual \
  --sd_scale 3.0 --sd_warmup_start 1.0 --sd_warmup_end 0.0 --sgf_sign paper \
  --dataset_name ring-a-bell --csv datasets/nudity-ring-a-bell.csv \
  --steps 4 --guidance_scale 3.5 --seed 42 \
  --metrics nudenet clip \
  --nudenet_path /path/to/nudenet_classifier_model.onnx \
  --save_path results/sgf

# STG (paper-faithful for nudity)
python eval_sgf_stg_meanflow.py \
  --method stg \
  --stg_nudenet_path checkpoints/nudenet/nudenet_640m.pt \
  --stg_lr 1.0 --stg_update_intervals 1-4 --stg_unsafe_conf 0.01 \
  --dataset_name ring-a-bell --csv datasets/nudity-ring-a-bell.csv \
  --metrics nudenet clip \
  --nudenet_path /path/to/nudenet_classifier_model.onnx \
  --save_path results/stg

# SAFREE-only
python eval_sgf_stg_meanflow.py \
  --method safree --safree_version safree --safree_alpha 0.01 \
  --dataset_name ring-a-bell --csv datasets/nudity-ring-a-bell.csv \
  --metrics nudenet clip \
  --nudenet_path /path/to/nudenet_classifier_model.onnx \
  --save_path results/safree

# Add SS to any: pass `--ss_version orig`
# Add SAFREE to any: pass `--safree_version safree`
```

### 5.2 Ours / VESFlow (driver: `eval_unified.py` or direct pipe.generate)

Single-config JSON (paper main table — VESFlow_str (stronger)):

```json
[
  {
    "risk_threshold": 0.3,
    "score_guide": true,
    "score_guide_vesflow_str": true,
    "score_guide_grad_target": "z",
    "score_guide_scorer": "laion_nsfw",
    "score_guide_kind": "sigmoid",
    "score_guide_min_t": 0.01,
    "score_guide_max_t": 0.95,
    "score_guide_factor_eps": 0.0,
    "score_guide_skip_first_step": false,
    "score_guide_divisor_max": 0.001,
    "score_guide_scale": 0.01,
    "safree": "none"
  }
]
```

For **VESFlow (basic)**, swap `"score_guide_vesflow_str": true` → `"score_guide_vesflow": true` and
use `"score_guide_scale": 3` (paper).

For **violence**, additionally pass `score_guide_laion_head_path` pointing at
`laion_trained_graphicharm_clipL.pt`.

---

## 6 · Method config cheat sheet (paper main table — MeanFlow 4-step)

| Method | Key hyperparameters |
|---|---|
| **VESFlow (basic)** | `score_guide_vesflow=True`, scorer=`laion_nsfw`, scale=3, t_max=0.95, divisor_max=0.001 |
| **VESFlow_str (stronger)** | `score_guide_vesflow_str=True`, scorer=`laion_nsfw`, scale=0.01, t_max=0.95, divisor_max=0.001 |
| **SGF** | scale=3.0, warmup 1.0→0.0, sign=`paper`, unsafe_images=i2p subset |
| **STG-Nudity** | NudeNet-YOLO 640m, lr=1.0, intervals=1-4, conf=0.01 |
| **STG-Violence** | CLIP scorer (`scorers.CLIPVelocityScorer`), texts=`{"violence","bloody","gore"}`, lr=1.0, intervals=1-4 |
| **SAFREE** | `safree_version="safree"`, alpha=0.01 |
| **Semantic Surgery** | `ss_version="orig"`, gamma=0.02, beta=-0.06, alpha_threshold=0.5 |

`risk_threshold=0.3` gates Ours-style score guidance to risky prompts only
(set 0.0 to apply unconditionally).

---

## 7 · Evaluation utilities

```bash
# Q16 inappropriate (violence/gore) ASR & TR on a directory of PNGs
python eval_q16_asr_tr.py results/my_method  \
    --threshold 0.6 --prompt_path checkpoints/q16/prompts.p

# MLLM (LLaVA-1.5-7B) yes/no cross-eval
python eval_mllm_safety.py --img_dirs results/method_A results/method_B \
    --device cuda:0
```

NudeNet ASR/TR is computed inline by `eval_sgf_stg_meanflow.py` when
`--metrics nudenet clip` is requested.

---

## 8 · Citation

```bibtex
@inproceedings{vesflow2026,
  title={VESFlow: Velocity Score-Based Guidance for Safe Few-Step Generation},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
```
