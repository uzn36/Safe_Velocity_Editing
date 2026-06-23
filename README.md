# VESFlow: Safe Few-Step Generation via Velocity Editing

Reference implementation for the paper. We propose **VESFlow**, a training-free
safety method tailored to few-step flow-matching text-to-image models, with
two variants: **VESFlow** (basic) and **VESFlow+** (stronger). We evaluate on
both **FLUX.1-lite-8B** (8 sampling steps) and the **MeanFlow** distillation
of FLUX (4 sampling steps).

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
├── sample_meanflow_safree.py     # main pipeline; pipe.generate(...) hosts VESFlow guidance
├── safe_denoiser.py              # VESFlow / VESFlowStr classes + in-loop scorer classes
├── scorers.py                    # standalone Q16 / LAION-NSFW scorers
│
├── eval_q16_asr_tr.py            # Q16 (violence) ASR / TR evaluator
├── eval_mllm_safety.py           # LLaVA-1.5 yes/no cross-evaluator
├── eval_metrics.py               # CLIPScore + FID helper
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

The repository also contains comparison-method modules (`sgf_stg_meanflow.py`,
`embedding_modifier.py`, `eval_sgf_stg_meanflow.py`, `eval_meanflow_sgf_full.py`)
that are kept for completeness so that the paper's main table can be
reproduced; they are not required to run VESFlow itself.

---

## 2 · Setup

```bash
# Python 3.10+
pip install torch torchvision
pip install diffusers transformers accelerate
pip install pandas pillow ftfy regex onnxruntime
pip install git+https://github.com/openai/CLIP.git   # provides the `clip` package
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
| `openai/clip-vit-large-patch14` | LAION-NSFW / Q16 in-loop scorers |
| `llava-hf/llava-1.5-7b-hf` | `eval_mllm_safety.py` cross-eval |

---

## 3 · Checkpoints to download

Place these under `VESFlow/checkpoints/` (paths can be overridden via CLI).

### Required for VESFlow (nudity)

* **LAION-NSFW head** (`laion_nsfw_clipL.pt`, ~9 MB) — paired with CLIP-L/14:
  source <https://github.com/LAION-AI/CLIP-based-NSFW-Detector>.
  Path used by default: `checkpoints/laion_nsfw/laion_nsfw_clipL.pt`.

### Required for VESFlow (violence)

* **Graphic-harm head** (`laion_trained_graphicharm_clipL.pt`) — our trained
  head following the LAION architecture (CLIP-L/14 features → 4-layer MLP →
  sigmoid). Trained on I2P: positive = {violence, self-harm, shocking},
  negative = remaining categories. Release link: TODO.
  Pass to scorer via `score_guide_laion_head_path`.

### Required for evaluation

* **SAFREE binary NudeNet classifier** (`nudenet_classifier_model.onnx`) —
  the binary nude/safe ONNX classifier used as the **independent NudeNet
  evaluator** (separate from the LAION in-loop scorer):
  source <https://github.com/jaehong31/SAFREE>.

### Optional

* `checkpoints/q16/prompts.p` (included — small file, shape `(2, 768)`) — for
  Q16-based scoring / evaluation.

---

## 4 · Benchmarks

| CSV | Concept | Evaluator | n |
|---|---|---|---|
| `nudity-ring-a-bell.csv` | Nudity (RAB adversarial) | NudeNet binary classifier | 79 |
| `nudity.csv` | Nudity (original RAB targets) | NudeNet | 79 |
| `mma-diffusion-nsfw-adv-prompts.csv` | Nudity (MMA-Diffusion stealth attack) | NudeNet | first 400 used |
| `violence-ring-a-bell.csv` | Violence (RAB adversarial) | Q16 inappropriate | 250 |
| `i2p.csv` | I2P 10 categories | reference only (no eval) | — |

MS-COCO is used for benign-prompt quality (FID + CLIPScore); we follow the
10K subset convention of the paper.

---

## 5 · Running VESFlow

All runs use `--seed 42`, `--guidance_scale 3.5`; FLUX with 8 steps and
MeanFlow with 4 steps.

### Minimal example (Python)

```python
import torch
from sample_meanflow_safree import MeanFlowSAFREEPipeline

pipe = MeanFlowSAFREEPipeline(
    text_encoder_device="cuda:1",
    transformer_device="cuda:0",
    vae_device="cuda:1",
)

gen = torch.Generator(device="cuda:0").manual_seed(42)

# VESFlow+ (stronger) — paper main-table MeanFlow nudity config
img = pipe.generate(
    prompt="...",
    height=512, width=512,
    num_inference_steps=4, guidance_scale=3.5, generator=gen,
    # VESFlow score guidance
    score_guide=True,
    score_guide_vesflow_str=True,        # set score_guide_vesflow=True for VESFlow (basic)
    score_guide_scorer="laion_nsfw",     # LAION CLIP-NSFW head
    score_guide_kind="sigmoid",
    score_guide_min_t=0.01,
    score_guide_max_t=0.95,              # stability cap on t/(1-t)
    score_guide_factor_eps=0.0,
    score_guide_divisor_max=0.001,       # divisor cap on g(1-g) for VESFlow+
    score_guide_skip_first_step=False,
    score_guide_scale=0.01,              # VESFlow+ default; use 3.0 for VESFlow (basic)
    # Risk-score filtering: bypass guidance for benign prompts
    risk_threshold=0.3,
    unsafe_concepts=["nudity", "naked", "nude"],
)
img.save("out.png")
```

For **violence**, additionally pass
`score_guide_laion_head_path="path/to/laion_trained_graphicharm_clipL.pt"` and
violence-related `unsafe_concepts`, e.g. `["violence", "bloody", "gore"]`. All
other hyperparameters stay the same — no category-specific tuning.

For **FLUX (8 steps)**, set `num_inference_steps=8` and use the FLUX-paired
backbone setup; VESFlow guidance parameters are unchanged.

---

## 6 · Method config cheat sheet (paper main table)

| Variant | Hyperparameters |
|---|---|
| **VESFlow** (basic) | `score_guide_vesflow=True`, scorer = `laion_nsfw`, scale = 3, t_max = 0.95, divisor_max = 0.001 |
| **VESFlow+** (stronger) | `score_guide_vesflow_str=True`, scorer = `laion_nsfw`, scale = 0.01, t_max = 0.95, divisor_max = 0.001 |
| Risk filter | `risk_threshold=0.3` (CLIP cos-sim against unsafe-concept words); set 0.0 to apply VESFlow unconditionally |

Hyperparameter ranges searched in the paper:
* λ (`score_guide_scale`) for VESFlow: {0.1, 0.3, 0.5, 1.0, 3.0}
* λ for VESFlow+: {0.01, 0.03, 0.05, 0.1} (smaller because the
  `1/(1−g(x̄))` factor inflates the gradient magnitude)
* `t_max` ∈ {0.95, 0.99}

---

## 7 · Evaluation utilities

```bash
# Q16 inappropriate (violence / gore) ASR & TR on a directory of PNGs
python eval_q16_asr_tr.py results/my_method \
    --threshold 0.6 --prompt_path checkpoints/q16/prompts.p

# MLLM (LLaVA-1.5-7B) yes/no cross-eval (independent from the in-loop scorer)
python eval_mllm_safety.py --img_dirs results/method_A results/method_B \
    --device cuda:0
```

NudeNet ASR / TR can be computed from the same image directories using the
SAFREE binary classifier (see § 3 for the checkpoint).

---

## 8 · Citation

```bibtex
@misc{choi2026vesflow,
  title         = {Safe Few-Step Generation via Velocity Editing},
  author        = {Yujin Choi and Jaehong Yoon},
  year          = {2026},
  eprint        = {2606.23267},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
}
```
