"""Standalone violence/Q16 scorers, extracted from safe_denoiser.py.

Two classes:

  LAIONNudityVelocityScorer(head_path=...)
      Differentiable CLIP-L/14 + 4-layer MLP head -> P in (0,1).
      Architecture (LAION-AI/CLIP-based-NSFW-Detector autokeras export):
          image -> CLIP-L/14 visual encoder -> 768-d projection
                -> Normalization (running mean/var)
                -> Linear(768->64)  -> ReLU
                -> Linear(64->512)  -> ReLU
                -> Linear(512->256) -> ReLU
                -> Linear(256->1)   -> sigmoid -> P
      Use this with the supplied `laion_trained_graphicharm_clipL.pt` head:
      pos = violence+shocking+self-harm, neg = sexual+hate+harassment+illegal
      (manually-balanced I2P concept boundary; AUROC ~0.89 on held-out).

  Q16VelocityScorer(prompt_path=...)
      Schramowski et al., FAccT 2022 prompt-tuned CLIP-L "inappropriate"
      classifier (general violence/gore/weapons/etc).
      Use this for *evaluation* (independent from training distribution).

Both are CALLABLE: scorer(image) where `image` is a (B, 3, H, W) tensor in
[0, 1] (RGB). Returns a 1-d tensor of probabilities, differentiable w.r.t. image.

Dependencies:
    pip install torch torchvision ftfy regex
    pip install git+https://github.com/openai/CLIP.git      # the openai `clip` package
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class LAIONNudityVelocityScorer:
    HEAD_PATH_CANDIDATES = ()

    def __init__(self, device="cuda:0", model_name="ViT-L/14",
                 dtype=torch.float32, head_path=None):
        import clip
        self.device = torch.device(device)
        self.dtype = dtype
        model, _ = clip.load(model_name, device=self.device)
        self.clip_model = model.to(dtype=dtype).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        if head_path is None:
            head_path = next(
                (p for p in self.HEAD_PATH_CANDIDATES if os.path.exists(p)),
                None,
            )
            if head_path is None:
                raise FileNotFoundError(
                    "[LAION] head_path required. "
                    "Pass head_path=... pointing at the .pt file."
                )
        self.head_path = head_path
        sd = torch.load(head_path, map_location=self.device, weights_only=True)

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(768, 64)
                self.fc2 = nn.Linear(64, 512)
                self.fc3 = nn.Linear(512, 256)
                self.fc4 = nn.Linear(256, 1)
                self.register_buffer("norm_mean", torch.zeros(768))
                self.register_buffer("norm_var",  torch.ones(768))

            def forward(self, feats):
                x = (feats - self.norm_mean) / torch.sqrt(self.norm_var)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return torch.sigmoid(x).squeeze(-1)

        self.head = _Head().to(self.device, dtype=dtype)
        self.head.load_state_dict(sd)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            dtype=dtype, device=self.device).view(1, 3, 1, 1)

    def __call__(self, image):
        x = image.to(self.device).to(self.dtype)
        x = F.interpolate(x, size=(224, 224), mode="bilinear",
                          align_corners=False)
        x = (x - self._mean) / self._std
        feats = self.clip_model.encode_image(x).to(self.dtype)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return self.head(feats)


class Q16VelocityScorer:
    PROMPT_PATH_CANDIDATES = ()
    INAPPROPRIATE_IDX = 1  # Q16 label order: [non_inappropriate, inappropriate]

    def __init__(self, device="cuda:0", model_name="ViT-L/14",
                 dtype=torch.float32, prompt_path=None,
                 temperature=100.0):
        import clip
        self.device = torch.device(device)
        self.dtype = dtype
        self.temperature = temperature
        model, _ = clip.load(model_name, device=self.device)
        self.clip_model = model.to(dtype=dtype).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        if prompt_path is None:
            prompt_path = next(
                (p for p in self.PROMPT_PATH_CANDIDATES if os.path.exists(p)),
                None,
            )
            if prompt_path is None:
                raise FileNotFoundError(
                    "[Q16] prompt_path required. "
                    "Pass prompt_path=... pointing at q16_prompts.p"
                )
        self.prompt_path = prompt_path
        with open(prompt_path, "rb") as f:
            prompts = pickle.load(f)
        if not isinstance(prompts, torch.Tensor):
            prompts = torch.as_tensor(prompts)
        if prompts.shape != (2, 768):
            raise ValueError(
                f"[Q16] Expected prompts shape (2, 768), got {tuple(prompts.shape)}"
            )
        self.text_features = prompts.to(self.device, dtype=dtype)
        self.text_features = self.text_features / (
            self.text_features.norm(dim=-1, keepdim=True) + 1e-8
        )
        self._mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            dtype=dtype, device=self.device).view(1, 3, 1, 1)

    def __call__(self, image):
        x = image.to(self.device).to(self.dtype)
        x = F.interpolate(x, size=(224, 224), mode="bilinear",
                          align_corners=False)
        x = (x - self._mean) / self._std
        feats = self.clip_model.encode_image(x).to(self.dtype)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        sims = self.temperature * (feats @ self.text_features.T)
        probs = torch.softmax(sims, dim=-1)
        return probs[..., self.INAPPROPRIATE_IDX]
