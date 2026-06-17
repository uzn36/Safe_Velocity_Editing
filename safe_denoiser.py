"""
Safe Denoiser / SGF for FLUX-based models (sCM, MeanFlow, standard Flux).

Implements denoising-step-level safety guidance:
  - Safe Denoiser (NeurIPS 2025): kernel-weighted repulsion from unsafe reference images
  - SGF (ICLR 2026 Oral): MMD gradient-based repulsion

Works alongside embedding_modifier.py (embedding-level) via callback_on_step_end.

Usage:
    from safe_denoiser import SafeDenoiserForFLUX

    sd = SafeDenoiserForFLUX(vae=pipe.vae, device="cuda:0")
    sd.load_unsafe_images("./unsafe_refs/", image_size=512)

    callback, cb_inputs = sd.get_callback(
        method="sgf",       # "safe_denoiser" or "sgf"
        scale=0.015,
        warmup_start=1,   # fraction of total steps (early = high noise)
        warmup_end=0.0,
    )

    result = pipeline(
        ...,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=cb_inputs,
    )
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================================================
# Repellency Methods (operate on latent x0_hat)
# ============================================================================

class SafeDenoiserRepellency:
    """
    Safe Denoiser (NeurIPS 2025): kernel-weighted empirical denoiser.

    x0_unsafe = Σ w_i * y_i  where w_i = k(x0_hat, y_i) / Σ k(x0_hat, y_j)
    x0_safe = x0_hat - scale * x0_unsafe

    Matches original: no input normalization, raw latents for kernel distance
    and weighted sum (repellency_methods_fast_sdv3.py:242-270).
    """

    def __init__(self, ref_latents, scale=1.0, sigma=1.0, epsilon=1e-8):
        self.ref_latents = ref_latents  # [N, C, H, W]
        self.scale = scale
        self.sigma = sigma
        self.epsilon = epsilon

    def compute(self, x0_hat):
        B = x0_hat.shape[0]
        ref = self.ref_latents.to(x0_hat.device, dtype=x0_hat.dtype)
        N = ref.shape[0]
        C, H, W = ref.shape[1], ref.shape[2], ref.shape[3]
        D = C * H * W

        # Flatten — NO normalization (matches original)
        x_flat = x0_hat.reshape(B, -1)       # [B, D]
        ref_flat = ref.reshape(N, -1)         # [N, D]

        ones = torch.ones(N, 1, device=ref.device, dtype=ref.dtype)

        # Kernel: exp(-||x - y|| / (2σ²)) — original uses L2 distance
        dist = torch.cdist(x_flat.unsqueeze(0), ref_flat.unsqueeze(0))[0]  # [B, N]
        kernel = (-dist.unsqueeze(-1).expand(B, N, D + 1) / (2.0 * self.sigma ** 2)).exp()
        ref_aug = torch.cat([ref_flat, ones], dim=1).unsqueeze(0)  # [1, N, D+1]
        weighted = (kernel * ref_aug).sum(dim=1)  # [B, D+1]

        denominator = weighted[:, -1:] + self.epsilon  # [B, 1]
        numerator = weighted[:, :-1]  # [B, D]

        negative_score = (numerator / denominator).reshape(B, C, H, W)

        x0_safe = x0_hat - self.scale * negative_score
        return x0_safe
    
    
class SafeDenoiserRepellency:
    """
    Safe Denoiser (NeurIPS 2025): kernel-weighted empirical denoiser.

    x0_unsafe = Σ w_i * y_i  where w_i = k(x0_hat, y_i) / Σ k(x0_hat, y_j)
    x0_safe = x0_hat - scale * x0_unsafe

    Matches original: no input normalization, raw latents for kernel distance
    and weighted sum (repellency_methods_fast_sdv3.py:242-270).
    """

    def __init__(self, ref_latents, scale=1.0, sigma=1.0, epsilon=1e-8):
        self.ref_latents = ref_latents  # [N, C, H, W]
        self.scale = scale
        self.sigma = sigma
        self.epsilon = epsilon

    def compute(self, x0_hat):
        B = x0_hat.shape[0]
        ref = self.ref_latents.to(x0_hat.device, dtype=x0_hat.dtype)
        N = ref.shape[0]
        C, H, W = ref.shape[1], ref.shape[2], ref.shape[3]
        D = C * H * W

        # Flatten — NO normalization (matches original)
        x_flat = x0_hat.reshape(B, -1)       # [B, D]
        ref_flat = ref.reshape(N, -1)         # [N, D]

        ones = torch.ones(N, 1, device=ref.device, dtype=ref.dtype)

        # Kernel: exp(-||x - y|| / (2σ²)) — original uses L2 distance
        dist = torch.cdist(x_flat.unsqueeze(0), ref_flat.unsqueeze(0))[0]  # [B, N]
        kernel = (-dist.unsqueeze(-1).expand(B, N, D + 1) / (2.0 * self.sigma ** 2)).exp()
        ref_aug = torch.cat([ref_flat, ones], dim=1).unsqueeze(0)  # [1, N, D+1]
        weighted = (kernel * ref_aug).sum(dim=1)  # [B, D+1]

        denominator = weighted[:, -1:] + self.epsilon  # [B, 1]
        numerator = weighted[:, :-1]  # [B, D]

        negative_score = (numerator / denominator).reshape(B, C, H, W)

        x0_safe = x0_hat - self.scale * negative_score
        return x0_safe



class SGFRepellency:
    """
    SGF (ICLR 2026 Oral): MMD gradient-based repellency.

    dK/dx = -2γ * K(x,y) * (x - y)  → points TOWARD refs
    x0_safe = x0_hat - scale * dK/dx  → subtract to REPEL

    Matches original: conditioning_1 does `x -= scale * dK_dX`
    (repellency_methods_grad_mmd.py + safer_pipeline.py).
    Uses adaptive RBF bandwidth from squared distances.
    """

    def __init__(self, ref_latents, scale=0.015, epsilon=1e-8, normalize=True):
        self.ref_latents = ref_latents
        self.scale = scale
        self.epsilon = epsilon
        self.normalize = normalize

    def _adaptive_gamma(self, x_flat, ref_flat, k=3):
        """Compute adaptive RBF bandwidth from k-NN squared distances (matches original)."""
        dist_sq = torch.cdist(x_flat, ref_flat).pow(2)  # squared, matching original
        topk_sq, _ = dist_sq.topk(min(k, dist_sq.shape[1]), dim=1, largest=False)
        mean_dist_sq = topk_sq.mean()
        eps = 0.05
        gamma = -math.log(eps) / (mean_dist_sq + self.epsilon)
        return gamma

    def compute(self, x0_hat):
        B, C, H, W = x0_hat.shape
        ref = self.ref_latents.to(x0_hat.device, dtype=x0_hat.dtype)
        N = ref.shape[0]

        x_flat = x0_hat.reshape(B, -1)     # [B, D]
        ref_flat = ref.reshape(N, -1)       # [N, D]

        # Adaptive bandwidth (squared distances, matching original)
        gamma = self._adaptive_gamma(x_flat, ref_flat)

        # Pairwise squared distances [B, N]
        dist_sq = torch.cdist(x_flat, ref_flat).pow(2)

        # RBF kernel K(x, y_i) = exp(-gamma * ||x - y_i||²)
        K = (-gamma * dist_sq).exp()  # [B, N]

        # Gradient: dK/dx = -2γ * K * (x - y), points toward refs
        diff = x_flat.unsqueeze(1) - ref_flat.unsqueeze(0)  # [B, N, D]
        dK_dX = (-2.0 * gamma * K.unsqueeze(-1) * diff).sum(dim=1)  # [B, D]

        if self.normalize:
            nrm = dK_dX.flatten(1).norm(p=2, dim=1, keepdim=True).clamp(min=self.epsilon)
            dK_dX = dK_dX / nrm

        # SUBTRACT to repel (dK/dx points toward refs → minus pushes away)
        x0_safe = x0_hat - self.scale * dK_dX.reshape(B, C, H, W)
        return x0_safe


def compute_risk_score_clip(prompt_text, unsafe_concepts, clip_model, clip_processor, device="cuda:0"):
    """
    Compute risk score using CLIP (better semantic separation than T5).
    Returns score in [0, 1] and per-concept similarities.
    """
    import torch
    texts = [prompt_text] + list(unsafe_concepts)
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = clip_model.text_model(**{k: v for k, v in inputs.items() if k != "pixel_values"})
        embeds = outputs.pooler_output  # [1+N, D]
    embeds = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-8)
    prompt_emb = embeds[0:1]   # [1, D]
    concept_embs = embeds[1:]  # [N, D]
    sim = (prompt_emb @ concept_embs.T).squeeze(0)  # [N]
    risk_score = sim.max().clamp(0, 1).item()
    return risk_score, sim


def compute_risk_score(prompt_embeds, ref_embeds, method="cosine"):
    """
    Concept-aware risk scoring (inspired by ConceptGuard, arxiv 2511.18780).

    Computes how close a prompt is to unsafe concepts.
    Returns a scalar risk score in [0, 1].

    Args:
        prompt_embeds: [B, S, D] or [B, D] prompt embedding
        ref_embeds: [N, D] unsafe concept embeddings (mean-pooled)
        method: "cosine" (default) or "projection"
    Returns:
        risk_score: float in [0, 1], max similarity to any unsafe concept
        per_concept: [N] similarity to each concept
    """
    # Mean-pool prompt if sequence
    if prompt_embeds.dim() == 3:
        p = prompt_embeds.float().mean(dim=1)  # [B, D]
    else:
        p = prompt_embeds.float()

    ref = ref_embeds.to(p.device, dtype=p.dtype)
    if ref.dim() == 3:
        ref = ref.mean(dim=1)  # [N, D]

    if method == "cosine":
        # Cosine similarity: s = <p/||p||, r/||r||>
        p_norm = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
        r_norm = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (p_norm @ r_norm.T).squeeze(0)  # [N]
        risk_score = sim.max().clamp(0, 1).item()
        return risk_score, sim
    elif method == "projection":
        # Projection norm: how much of prompt lies in unsafe subspace
        _, _, Vh = torch.linalg.svd(ref, full_matrices=False)
        r = min(ref.shape[0], ref.shape[1])
        V_r = Vh[:r, :].T  # [D, r]
        p_proj = p @ V_r @ V_r.T  # [B, D]
        ratio = p_proj.norm() / (p.norm() + 1e-8)
        return ratio.item(), None
    else:
        raise ValueError(f"Unknown method: {method}")


class CLIPSafetyGuidance:
    """
    CLIP-based safety guidance: predict x_0 at step t, decode to image,
    check CLIP similarity with unsafe concepts, backprop to c if unsafe.

    Applied as callback. Requires: transformer, VAE, CLIP (all differentiable).

    Flow at each step:
        c.requires_grad_(True) → transformer(x_t, c, t) → v → x_0_hat
        → VAE.decode(x_0_hat) → CLIP(image, unsafe_text) → score
        → if score > τ: ∇_c score → c' = c - lr · ∇_c
    """

    def __init__(self, clip_model, clip_processor, unsafe_texts,
                 threshold=0.3, lr=0.1, device="cuda:0"):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.unsafe_texts = unsafe_texts
        self.threshold = threshold
        self.lr = lr
        self.device = device

        # Pre-encode unsafe text
        with torch.no_grad():
            inputs = clip_processor(text=unsafe_texts, return_tensors="pt",
                                    padding=True, truncation=True).to(device)
            self.unsafe_text_embeds = clip_model.text_model(
                **{k: v for k, v in inputs.items() if k != "pixel_values"}
            ).pooler_output
            self.unsafe_text_embeds = self.unsafe_text_embeds / (
                self.unsafe_text_embeds.norm(dim=-1, keepdim=True) + 1e-8)

    def compute_clip_score(self, image_tensor):
        """Compute max cosine sim between image and unsafe texts (differentiable)."""
        # image_tensor: [B, 3, H, W] in [0, 1]
        # Resize to CLIP input (224x224)
        image_resized = F.interpolate(image_tensor, size=(224, 224), mode="bilinear")
        # Normalize with CLIP stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device).view(1, 3, 1, 1)
        image_norm = (image_resized - mean) / std
        # CLIP vision forward
        image_embeds = self.clip_model.vision_model(image_norm).pooler_output
        image_embeds = self.clip_model.visual_projection(image_embeds)
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        # Cosine sim with unsafe texts
        sim = (image_embeds @ self.unsafe_text_embeds.T)  # [B, N_concepts]
        return sim.max(dim=-1).values  # [B] — max similarity

    def correct_embedding(self, prompt_embeds, x_t, sigma_t, transformer, vae,
                          pooled_prompt_embeds=None, text_ids=None,
                          latent_image_ids=None, guidance=None, timestep=None,
                          transformer_extra_kwargs=None):
        """
        One-step correction: predict x_0, check safety, update c if unsafe.

        Args:
            prompt_embeds: [B, S, D]
            x_t: [B, S_lat, D_lat] packed latents
            sigma_t: current sigma
            transformer: the diffusion transformer
            vae: VAE for decoding
        Returns:
            corrected prompt_embeds
        """
        B = prompt_embeds.shape[0]

        with torch.enable_grad():
            c = prompt_embeds.detach().clone().requires_grad_(True)

            # Transformer forward with grad on c
            extra = transformer_extra_kwargs or {}
            noise_pred = transformer(
                hidden_states=x_t,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=c,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
                **extra,
            )[0]

            # x_0 prediction (flow matching)
            x0_hat = x_t - sigma_t * noise_pred

            # Unpack and decode
            # For now assume packed FLUX latents
            lat_shape = x0_hat.shape
            if len(lat_shape) == 3:
                B_l, S, D = lat_shape
                h = w = int(math.sqrt(S))
                c_lat = D // 4
                x0_unpack = x0_hat.view(B_l, h, w, c_lat, 2, 2).permute(0, 3, 1, 4, 2, 5)
                x0_unpack = x0_unpack.reshape(B_l, c_lat, h * 2, w * 2)
            else:
                x0_unpack = x0_hat

            # VAE decode (move x0 to VAE's device; autograd tracks device transfer)
            vae_dev = next(vae.parameters()).device
            x0_scaled = (x0_unpack / vae.config.scaling_factor) + vae.config.shift_factor
            x0_scaled = x0_scaled.to(vae_dev, dtype=vae.dtype)
            image = vae.decode(x0_scaled).sample
            image = (image / 2 + 0.5).clamp(0, 1)  # [B, 3, H, W]

            # CLIP safety score (move image to CLIP device)
            image_for_clip = image.to(self.device).float()
            score = self.compute_clip_score(image_for_clip)  # [B]

        # Check if unsafe
        is_unsafe = score > self.threshold
        if is_unsafe.any():
            # Compute gradient
            grads = torch.autograd.grad(score.sum(), c, retain_graph=False)[0]
            grads_norm = grads / (grads.norm(dim=(1, 2), keepdim=True) + 1e-8)

            # Update: move c away from unsafe direction
            corrected = prompt_embeds.clone()
            for b in range(B):
                if is_unsafe[b]:
                    corrected[b] = prompt_embeds[b] - self.lr * grads_norm[b]

            return corrected.to(prompt_embeds.dtype), score.detach()
        else:
            return prompt_embeds, score.detach()


class NudeNetVSG:
    """
    NudeNet-based Velocity Safety Guidance.

    Uses PyTorch NudeNet (YOLO-based) to compute ∇_v NudeNet_score,
    then modifies velocity to steer away from unsafe content.

    Flow:
        v.requires_grad_(True) → x_0 = x_t - σ·v → VAE.decode(x_0) → image
        → NudeNet(image) → score → ∇_v score → v' = v - ρ·∇_v
    """

    def __init__(self, nudenet_model, vae, scale=1.0, threshold=0.0):
        """
        Args:
            nudenet_model: PyTorch NudeNet (with safety_score method)
            vae: VAE decoder
            scale: guidance strength
            threshold: only apply if score > threshold
        """
        self.nudenet = nudenet_model
        self.vae = vae
        self.scale = scale
        self.threshold = threshold

    def correct_velocity(self, v_pred, x_t, sigma_t):
        """
        Compute NudeNet gradient on velocity and return corrected v.

        Args:
            v_pred: [B, C, H, W] or packed — velocity prediction
            x_t: same shape — current latent
            sigma_t: current noise level
        Returns:
            v_corrected, score
        """
        with torch.enable_grad():
            v = v_pred.detach().clone().requires_grad_(True)

            # x_0 prediction
            x0_hat = x_t - sigma_t * v

            # VAE decode
            x0_scaled = (x0_hat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(x0_scaled.to(self.vae.dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)

            # NudeNet score (differentiable)
            score, masks = self.nudenet.safety_score(image)

        if score.sum() > self.threshold:
            grads = torch.autograd.grad(score.sum(), v, retain_graph=False)[0]
            grads_norm = grads / (grads.norm() + 1e-8)
            v_corrected = v_pred - self.scale * grads_norm
            return v_corrected.to(v_pred.dtype), score.detach()
        else:
            return v_pred, score.detach()


# ============================================================================
# Iterative Velocity Safety Guidance (inner-loop refine)
# ============================================================================

class VESFlow:
    """Image-space (z) gradient variant of ScoreBasedGuidance3.

    Same logit-domain g · ∇h = ∇g / (1 − g) as v3, but the autograd is taken
    w.r.t. the decoded image z (not the latent). The image-space gradient is
    converted to a latent delta via VAE re-encode, identical to v2Z.

        z         = decode(x̂_0)
        h(z)      = log[g(z) / (1 − g(z))]
        ∇_z h     = autograd
        g · ∇_z h = ∇_z g / (1 − g)            (logit-domain image-space grad)
        z_new     = clip(z − λ·(t/(1−t))·g·∇_z h, 0, 1)
        Δx̂_0    = encode(z_new) − encode(z)
        u_new     = u − Δx̂_0 / t
    """

    def __init__(self, vae, scorer, scale=1.0,
                 kind="sigmoid", sigma_data=None,
                 min_t=0.01, max_t=0.95, normalize_grad=False,
                 latent_packed=True, verbose=False, divisor_max=0.01,
                 grad_blur_sigma=0.0, grad_blur_kernel=5,
                 factor_eps=0.0,
                 grad_pool_size=1, grad_pool_mode="bilinear",
                 factor_cap=0.0):
        assert kind in ("sigmoid", "raw", "binary")
        self.vae = vae
        self.scorer = scorer
        self.scale = float(scale)
        self.kind = kind
        self.sigma_data = sigma_data
        self.min_t = float(min_t)
        self.max_t = float(max_t)
        self.normalize_grad = bool(normalize_grad)
        self.latent_packed = bool(latent_packed)
        self.verbose = bool(verbose)
        self.divisor_max = float(divisor_max)
        self.grad_blur_sigma = float(grad_blur_sigma)
        self.grad_blur_kernel = int(grad_blur_kernel)
        # When > 0, factor = t / (1 - t + factor_eps): allows max_t=1.0 with
        # finite factor capped at 1/factor_eps.
        self.factor_eps = float(factor_eps)
        # Low-frequency restriction on the (image-space) gradient: avg_pool the
        # gradient down by `grad_pool_size`, then interpolate back. Forces the
        # update to act at coarser-than-pixel scales — defeats per-pixel
        # adversarial noise that fools the scorer without changing semantics.
        self.grad_pool_size = int(grad_pool_size)
        self.grad_pool_mode = str(grad_pool_mode)
        # Hard upper-bound on the temporal amplifier: factor = min(t/(1-t), factor_cap).
        # Default 0 ⇒ disabled (use factor_eps formulation instead).
        # When > 0, replaces factor_eps and prevents the t→1 blow-up.
        self.factor_cap = float(factor_cap)

    def _unpack(self, x_hat_0):
        if not self.latent_packed:
            return x_hat_0
        B, S, D = x_hat_0.shape
        h = w = int(math.sqrt(S))
        c = D // 4
        x = x_hat_0.view(B, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, c, h * 2, w * 2)

    def _pack(self, x_unpacked):
        if not self.latent_packed:
            return x_unpacked
        B, C, H, W = x_unpacked.shape
        return (x_unpacked.reshape(B, C, H // 2, 2, W // 2, 2)
                          .permute(0, 2, 4, 1, 3, 5)
                          .reshape(B, (H // 2) * (W // 2), C * 4))

    def _decode(self, x_hat_0_unpacked):
        if self.sigma_data is not None:
            x_hat_0_unpacked = x_hat_0_unpacked / float(self.sigma_data)
        vae_dev = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        x = (x_hat_0_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x = x.to(vae_dev, dtype=vae_dtype)
        img = self.vae.decode(x).sample
        img = (img / 2 + 0.5).clamp(0, 1).float()
        return img

    def _encode(self, img_01):
        vae_dev = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        x_raw = (img_01 - 0.5) * 2.0
        x_raw = x_raw.to(vae_dev, dtype=vae_dtype)
        lat = self.vae.encode(x_raw).latent_dist.mean
        x = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        if self.sigma_data is not None:
            x = x * float(self.sigma_data)
        return x

    def apply(self, u, x_t, t, r=0.0):
        if t < self.min_t:
            return u
        del r
        dtype, dev = u.dtype, u.device
        t_clamped = min(float(t), self.max_t)
        # factor = min(t/(1-t), factor_cap) when factor_cap > 0, else t/(1-t+factor_eps).
        if self.factor_cap > 0:
            factor = min(t_clamped / max(1.0 - t_clamped, 1e-8), self.factor_cap)
        else:
            factor = t_clamped / max(1.0 - t_clamped + self.factor_eps, 1e-8)

        x_hat_0_packed = x_t.detach() - t * u.detach()
        x_hat_0_unpacked = self._unpack(x_hat_0_packed)

        with torch.enable_grad():
            img = self._decode(x_hat_0_unpacked).requires_grad_(True)
            if self.kind == "binary":
                logits_u, logits_s = self.scorer(img)
                target = (logits_u - logits_s).reshape(-1).sum()
                grads_z = torch.autograd.grad(target, img)[0]
                log_ratio_val = float(target.item())
                g_val = None
            else:
                # Pure logit-domain: h = log(g/(1-g)), autograd ∇_z h, multiply by g.
                # g · ∇_z h ≡ ∇_z g / (1 − g)
                g = self.scorer(img)
                if g.ndim > 1:
                    g = g.reshape(g.shape[0], -1).sum(dim=-1)
                g_safe = g.clamp(1e-4, 1 - 1e-4)
                h = torch.log(g_safe) - torch.log(1.0 - g_safe)
                grad_h = torch.autograd.grad(h.sum(), img)[0]
                g_val = g.detach()
                g_factor = g_val.clamp(1e-4, 1 - 1e-4).reshape(
                    -1, *([1] * (grad_h.ndim - 1)))
                grads_z = g_factor.to(grad_h.device, dtype=grad_h.dtype) * grad_h
                log_ratio_val = float(h.sum().item())

        grad_norm = float(grads_z.flatten(1).norm(dim=1).mean())
        if self.normalize_grad:
            nrm = grads_z.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
            grads_z = grads_z / nrm.view(-1, *([1] * (grads_z.ndim - 1)))

        if self.grad_blur_sigma > 0:
            grads_z = gaussian_blur_packed_latent(
                grads_z, kernel_size=self.grad_blur_kernel,
                sigma=self.grad_blur_sigma, latent_packed=False
            ).to(grads_z.dtype)

        if self.grad_pool_size > 1:
            # Pool→unpool to suppress per-pixel adversarial noise: forces the
            # gradient to act on patches of size grad_pool_size×grad_pool_size.
            _g_dtype = grads_z.dtype
            _g_f = grads_z.float()
            _pooled = F.avg_pool2d(_g_f, kernel_size=self.grad_pool_size,
                                   stride=self.grad_pool_size)
            if self.grad_pool_mode == "nearest":
                _up = F.interpolate(_pooled, size=_g_f.shape[-2:], mode="nearest")
            else:
                _up = F.interpolate(_pooled, size=_g_f.shape[-2:],
                                    mode=self.grad_pool_mode, align_corners=False)
            grads_z = _up.to(_g_dtype)

        with torch.no_grad():
            img_orig = self._decode(x_hat_0_unpacked).detach()
            img_new = (img_orig - (self.scale * factor) * grads_z.to(img_orig.dtype)).clamp(0, 1)
            x_hat_0_new_unpacked = self._encode(img_new)
            x_hat_0_orig_unpacked = self._encode(img_orig)
            dx_unpacked = x_hat_0_new_unpacked - x_hat_0_orig_unpacked
            dx_packed = self._pack(dx_unpacked).to(dev, dtype=dtype)
            u_new = u - dx_packed / max(t, 1e-4)

        if self.verbose:
            extra = "" if g_val is None else f"  g={float(g_val.mean()):.4f}"
            cap_str = "" if self.factor_cap <= 0 else f"  factor_cap={self.factor_cap:.1f}"
            print(f"    [ScoreGuide3Z] t={t:.3f}  factor={factor:.3f}  "
                  f"log_ratio={log_ratio_val:.4f}  z_grad_norm={grad_norm:.4f}"
                  f"{extra}  eff_step≈{self.scale*factor:.3f}{cap_str}")

        return u_new.to(dtype)


class VESFlowStr:
    """Image-space (z) gradient variant of ScoreBasedGuidance4.

    Same pure logit gradient ∇h = ∇g / [g (1 − g)] as v4, but autograd is taken
    w.r.t. the decoded image z, then converted to a latent delta via VAE
    re-encode (same conversion as v2Z / v3Z).

        z       = decode(x̂_0)
        ∇_z h   = ∇_z g / [g (1 − g)]            (image-space, with divisor_max cap)
        z_new   = clip(z − λ·(t/(1−t))·∇_z h, 0, 1)
        Δx̂_0  = encode(z_new) − encode(z)
        u_new   = u − Δx̂_0 / t
    """

    def __init__(self, vae, scorer, scale=1.0,
                 kind="sigmoid", sigma_data=None,
                 min_t=0.01, max_t=0.95, normalize_grad=False,
                 latent_packed=True, verbose=False, divisor_max=0.01,
                 grad_blur_sigma=0.0, grad_blur_kernel=5,
                 factor_eps=0.0,
                 grad_pool_size=1, grad_pool_mode="bilinear",
                 factor_cap=0.0):
        assert kind in ("sigmoid", "raw", "binary")
        self.vae = vae
        self.scorer = scorer
        self.scale = float(scale)
        self.kind = kind
        self.sigma_data = sigma_data
        self.min_t = float(min_t)
        self.max_t = float(max_t)
        self.normalize_grad = bool(normalize_grad)
        self.latent_packed = bool(latent_packed)
        self.verbose = bool(verbose)
        self.divisor_max = float(divisor_max)
        self.grad_blur_sigma = float(grad_blur_sigma)
        self.grad_blur_kernel = int(grad_blur_kernel)
        # When > 0, factor = t / (1 - t + factor_eps): allows max_t=1.0 with
        # finite factor capped at 1/factor_eps.
        self.factor_eps = float(factor_eps)
        # Low-frequency restriction on the (image-space) gradient: avg_pool the
        # gradient down by `grad_pool_size`, then interpolate back. Forces the
        # update to act at coarser-than-pixel scales — defeats per-pixel
        # adversarial noise that fools the scorer without changing semantics.
        self.grad_pool_size = int(grad_pool_size)
        self.grad_pool_mode = str(grad_pool_mode)
        # Hard upper-bound on the temporal amplifier: factor = min(t/(1-t), factor_cap).
        # Default 0 ⇒ disabled (use factor_eps formulation instead).
        self.factor_cap = float(factor_cap)

    def _unpack(self, x_hat_0):
        if not self.latent_packed:
            return x_hat_0
        B, S, D = x_hat_0.shape
        h = w = int(math.sqrt(S))
        c = D // 4
        x = x_hat_0.view(B, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, c, h * 2, w * 2)

    def _pack(self, x_unpacked):
        if not self.latent_packed:
            return x_unpacked
        B, C, H, W = x_unpacked.shape
        return (x_unpacked.reshape(B, C, H // 2, 2, W // 2, 2)
                          .permute(0, 2, 4, 1, 3, 5)
                          .reshape(B, (H // 2) * (W // 2), C * 4))

    def _decode(self, x_hat_0_unpacked):
        if self.sigma_data is not None:
            x_hat_0_unpacked = x_hat_0_unpacked / float(self.sigma_data)
        vae_dev = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        x = (x_hat_0_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x = x.to(vae_dev, dtype=vae_dtype)
        img = self.vae.decode(x).sample
        img = (img / 2 + 0.5).clamp(0, 1).float()
        return img

    def _encode(self, img_01):
        vae_dev = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        x_raw = (img_01 - 0.5) * 2.0
        x_raw = x_raw.to(vae_dev, dtype=vae_dtype)
        lat = self.vae.encode(x_raw).latent_dist.mean
        x = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        if self.sigma_data is not None:
            x = x * float(self.sigma_data)
        return x

    def apply(self, u, x_t, t, r=0.0):
        if t < self.min_t:
            return u
        del r
        dtype, dev = u.dtype, u.device
        t_clamped = min(float(t), self.max_t)
        # factor = min(t/(1-t), factor_cap) when factor_cap > 0, else t/(1-t+factor_eps).
        if self.factor_cap > 0:
            factor = min(t_clamped / max(1.0 - t_clamped, 1e-8), self.factor_cap)
        else:
            factor = t_clamped / max(1.0 - t_clamped + self.factor_eps, 1e-8)

        x_hat_0_packed = x_t.detach() - t * u.detach()
        x_hat_0_unpacked = self._unpack(x_hat_0_packed)

        with torch.enable_grad():
            img = self._decode(x_hat_0_unpacked).requires_grad_(True)
            if self.kind == "binary":
                logits_u, logits_s = self.scorer(img)
                target = (logits_u - logits_s).reshape(-1).sum()
                grads_z = torch.autograd.grad(target, img)[0]
                log_ratio_val = float(target.item())
                g_val = None
            else:
                # Pure logit gradient: ∇_z h = ∇_z g / [g (1 − g)] with divisor cap
                g = self.scorer(img)
                if g.ndim > 1:
                    g = g.reshape(g.shape[0], -1).sum(dim=-1)
                grad_g = torch.autograd.grad(g.sum(), img)[0]
                g_val = g.detach()
                gw = g_val.clamp(1e-4, 1 - 1e-4)
                gw_factor = (gw * (1.0 - gw)).clamp(max=self.divisor_max).reshape(
                    -1, *([1] * (grad_g.ndim - 1)))
                grads_z = grad_g / gw_factor.to(grad_g.device, dtype=grad_g.dtype)
                log_ratio_val = float((torch.log(gw) - torch.log(1.0 - gw)).sum().item())

        grad_norm = float(grads_z.flatten(1).norm(dim=1).mean())
        if self.normalize_grad:
            nrm = grads_z.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
            grads_z = grads_z / nrm.view(-1, *([1] * (grads_z.ndim - 1)))

        if self.grad_blur_sigma > 0:
            grads_z = gaussian_blur_packed_latent(
                grads_z, kernel_size=self.grad_blur_kernel,
                sigma=self.grad_blur_sigma, latent_packed=False
            ).to(grads_z.dtype)

        if self.grad_pool_size > 1:
            # Pool→unpool to suppress per-pixel adversarial noise: forces the
            # gradient to act on patches of size grad_pool_size×grad_pool_size.
            _g_dtype = grads_z.dtype
            _g_f = grads_z.float()
            _pooled = F.avg_pool2d(_g_f, kernel_size=self.grad_pool_size,
                                   stride=self.grad_pool_size)
            if self.grad_pool_mode == "nearest":
                _up = F.interpolate(_pooled, size=_g_f.shape[-2:], mode="nearest")
            else:
                _up = F.interpolate(_pooled, size=_g_f.shape[-2:],
                                    mode=self.grad_pool_mode, align_corners=False)
            grads_z = _up.to(_g_dtype)

        with torch.no_grad():
            img_orig = self._decode(x_hat_0_unpacked).detach()
            img_new = (img_orig - (self.scale * factor) * grads_z.to(img_orig.dtype)).clamp(0, 1)
            x_hat_0_new_unpacked = self._encode(img_new)
            x_hat_0_orig_unpacked = self._encode(img_orig)
            dx_unpacked = x_hat_0_new_unpacked - x_hat_0_orig_unpacked
            dx_packed = self._pack(dx_unpacked).to(dev, dtype=dtype)
            u_new = u - dx_packed / max(t, 1e-4)

        if self.verbose:
            extra = "" if g_val is None else f"  g={float(g_val.mean()):.4f}"
            cap_str = "" if self.factor_cap <= 0 else f"  factor_cap={self.factor_cap:.1f}"
            print(f"    [ScoreGuide4Z] t={t:.3f}  factor={factor:.3f}  "
                  f"log_ratio={log_ratio_val:.4f}  z_grad_norm={grad_norm:.4f}"
                  f"{extra}  eff_step≈{self.scale*factor:.3f}{cap_str}")

        return u_new.to(dtype)


class EnsembleScorer:
    """Weighted average of multiple scorers (each must return [B] in compatible range).

    For ensemble of NudeNet 'max' (∈ [0,1]) and CLIP normalize='01' (∈ [0,1]):
        g_ens(image) = w_nudenet · g_nudenet + w_clip · g_clip
    Differentiable: autograd composes through each scorer.

    Args:
        scorers : list of base scorers (callable image → [B])
        weights : optional list of floats, must sum > 0; default = uniform 1/N each
    """

    def __init__(self, scorers, weights=None):
        self.scorers = list(scorers)
        if weights is None:
            weights = [1.0 / len(scorers)] * len(scorers)
        s = sum(weights)
        self.weights = [w / s for w in weights]

    def __call__(self, image):
        out = None
        for w, scorer in zip(self.weights, self.scorers):
            v = scorer(image)
            term = w * v
            out = term if out is None else out + term
        return out


def gaussian_blur_packed_latent(grads_packed, kernel_size=5, sigma=1.0,
                                 latent_packed=True):
    """Apply 2D spatial Gaussian blur to a latent gradient.

    Args:
        grads_packed  : [B, S, D] packed FLUX/MeanFlow latent gradient,
                        OR [B, C, H, W] unpacked.
        kernel_size   : odd int, default 5
        sigma         : Gaussian std in pixels (latent pixels)
        latent_packed : whether input is packed (True) or unpacked (False)
    """
    import torch.nn.functional as F
    if latent_packed:
        B, S, D = grads_packed.shape
        h = w = int(math.sqrt(S))
        c = D // 4
        x = grads_packed.view(B, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, c, h * 2, w * 2)
    else:
        x = grads_packed
    C = x.shape[1]
    half = kernel_size // 2
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - half
    k1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = k1d[:, None] * k1d[None, :]
    k2d = k2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)
    x_blur = F.conv2d(x, k2d, padding=half, groups=C)
    if latent_packed:
        B, C, H, W = x_blur.shape
        out = (x_blur.reshape(B, C, H // 2, 2, W // 2, 2)
                     .permute(0, 2, 4, 1, 3, 5)
                     .reshape(B, (H // 2) * (W // 2), C * 4))
        return out
    return x_blur


class SmoothedScorer:
    """Randomized-smoothing wrapper over a base scorer.

    Score(z) = E_{ε ~ N(0, σ²I)} [ base_scorer(clip(z + ε, 0, 1)) ]

    Approximated by `n_samples` Monte-Carlo draws. Differentiable end-to-end:
    autograd through the average gives a smoothed gradient that resists
    adversarial high-frequency directions in the image.

    Args:
        base_scorer : a callable image → [B] score.
        noise_sigma : Gaussian std of additive image noise. 0 disables smoothing.
        n_samples   : number of MC samples per call. =1 is "single noisy probe".
    """

    def __init__(self, base_scorer, noise_sigma=0.05, n_samples=4):
        self.base = base_scorer
        self.noise_sigma = float(noise_sigma)
        self.n_samples = int(n_samples)
        # forward attribute access for any callers that probe the underlying scorer
        for attr in ("device",):
            if hasattr(base_scorer, attr):
                setattr(self, attr, getattr(base_scorer, attr))

    def __call__(self, image):
        if self.noise_sigma <= 0 or self.n_samples <= 0:
            return self.base(image)
        # batch the N samples in one forward pass
        N = self.n_samples
        img_rep = image.repeat(N, *([1] * (image.ndim - 1)))
        noise = self.noise_sigma * torch.randn_like(img_rep)
        noisy = (img_rep + noise).clamp(0.0, 1.0)
        scores = self.base(noisy)            # [N*B] (or [N*B, ...] if multi-out)
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        B = image.shape[0]
        return scores.view(N, B, *scores.shape[1:]).mean(dim=0)  # [B, ...]


class CLIPVelocityScorer:
    """Differentiable CLIP cosine-sim scorer for images.

    Wraps the same CLIP flow as CLIPSafetyGuidance.compute_clip_score but packaged
    as a plain callable so IterativeVSG can plug arbitrary scorers in.
    """

    DEFAULT_SAFE_TEXTS = (
        "a photo of a clothed person",
        "a landscape photograph",
        "a still life",
        "an everyday scene",
    )

    def __init__(self, clip_model, clip_processor, unsafe_texts, device="cuda:0",
                 normalize="01", safe_texts=None):
        """normalize: how to map raw cosine sim ∈ [-1, 1] to a probability-like value.
            "01"       : (sim + 1) / 2                  ∈ [0, 1]   (linear stretch)
            "sigmoid"  : torch.sigmoid(sim)             ∈ (0, 1)   (still flat)
            "zeroshot" : softmax(logit_scale·[sim_unsafe, sim_safe])[0]  ∈ (0, 1)
                         True CLIP zero-shot classifier — sharp probability since
                         CLIP's learned logit_scale (~100) makes the softmax steep.
                         Requires safe_texts (or uses DEFAULT_SAFE_TEXTS).
            "raw"      : leave as-is                    ∈ [-1, 1]
        """
        self.clip_model = clip_model
        self.device = device
        self.normalize = normalize

        def _encode(texts):
            inputs = clip_processor(text=list(texts), return_tensors="pt",
                                    padding=True, truncation=True).to(device)
            txt_out = clip_model.text_model(
                **{k: v for k, v in inputs.items() if k != "pixel_values"}
            ).pooler_output
            txt_out = clip_model.text_projection(txt_out)
            return txt_out / (txt_out.norm(dim=-1, keepdim=True) + 1e-8)

        with torch.no_grad():
            self.unsafe_text_embeds = _encode(unsafe_texts)
            if normalize == "zeroshot":
                self.safe_text_embeds = _encode(
                    safe_texts if safe_texts else self.DEFAULT_SAFE_TEXTS)

    def __call__(self, image):
        # image: [B, 3, H, W] in [0, 1] on any device — move to CLIP device
        image = image.to(self.device)
        image_resized = F.interpolate(image, size=(224, 224), mode="bilinear")
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=self.device).view(1, 3, 1, 1)
        image_norm = (image_resized - mean) / std
        image_embeds = self.clip_model.vision_model(image_norm).pooler_output
        image_embeds = self.clip_model.visual_projection(image_embeds)
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        sim_unsafe = (image_embeds @ self.unsafe_text_embeds.T).max(dim=-1).values  # [B]
        if self.normalize == "zeroshot":
            sim_safe = (image_embeds @ self.safe_text_embeds.T).max(dim=-1).values  # [B]
            logit_scale = self.clip_model.logit_scale.exp()
            logits = torch.stack([sim_unsafe, sim_safe], dim=-1) * logit_scale     # [B, 2]
            return torch.softmax(logits, dim=-1)[:, 0]                              # [B] p_unsafe
        if self.normalize == "01":
            return (sim_unsafe + 1.0) * 0.5
        if self.normalize == "sigmoid":
            return torch.sigmoid(sim_unsafe)
        return sim_unsafe


class NudeNetYOLOVelocityScorer:
    """Differentiable YOLO-NudeNet scorer.

    Args:
        yolo_checkpoint : path to YOLO checkpoint
        device          : target CUDA device
        reduce          : how to combine per-image detections into a scalar score:
                          'sum' → sum of unsafe-class confidences (legacy, can exceed 1)
                          'max' → max single-detection confidence (∈ [0, 1], probability-like)
    """

    UNSAFE_CLASSES = (3, 4, 6, 14)  # FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED,
                                    # ANUS_EXPOSED, MALE_GENITALIA_EXPOSED

    def __init__(self, yolo_checkpoint, device="cuda:0", reduce="max",
                 mean_threshold=0.0):
        import importlib.util, sys
        stg_path = "/workspace/STG/nudenet/nudenet.py"
        spec = importlib.util.spec_from_file_location("stg_nudenet", stg_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["stg_nudenet"] = module
        spec.loader.exec_module(module)
        NudeDetector_YOLO = module.NudeDetector_YOLO
        self.detector = NudeDetector_YOLO(yolo_checkpoint, device=device)
        self.device = torch.device(device)
        self.reduce = reduce
        # mean_threshold > 0 only used with reduce='mean': drop confs ≤ threshold
        # before averaging, so noisy marginal detections don't dilute the score.
        self.mean_threshold = float(mean_threshold)
        try:
            self.detector.model.to(self.device)
            self.detector.model.model.to(self.device)
            self.detector.predictor.model.to(self.device)
        except AttributeError:
            pass

    def __call__(self, image):
        image = image.to(self.device).float()
        if self.reduce == "sum":
            scores, _ = self.detector.safety_score(image)
            return scores  # [B]
        # 'max' or 'mean' reduction over UNSAFE_CLASSES detections
        from ultralytics.utils import ops
        det = self.detector
        img = det.resize_transform2(image).clamp(0, 1)
        output = det.model.model(img)
        preds = ops.non_max_suppression(
            output, det.predictor.args.conf, det.predictor.args.iou,
            det.predictor.args.classes, det.predictor.args.agnostic_nms,
            max_det=det.predictor.args.max_det,
            nc=len(det.model.names),
            end2end=getattr(det.predictor.args.model, "end2end", False),
            rotated=det.predictor.args.task == "obb",
        )
        unsafe_cls = torch.tensor(self.UNSAFE_CLASSES, device=img.device)
        # Keep the autograd graph connected to `image` even when an image has
        # zero unsafe detections (otherwise scores becomes a constant tensor
        # with no grad_fn and torch.autograd.grad fails).
        scores = torch.zeros((len(preds),), device=img.device) + image.flatten().sum() * 0.0
        for idx, pred in enumerate(preds):
            mask = torch.isin(pred[:, 5], unsafe_cls)
            if not mask.any():
                continue
            confs = pred[mask, 4]
            if self.reduce == "max":
                scores[idx] = confs.max()
            elif self.reduce == "mean":
                if self.mean_threshold > 0:
                    above = confs[confs > self.mean_threshold]
                    # If no box passes τ, fall back to max(confs) so the autograd
                    # graph stays connected (gradient → 0 only when the scorer
                    # itself stops responding, not when threshold filter empties).
                    scores[idx] = above.mean() if above.numel() > 0 else confs.max()
                else:
                    scores[idx] = confs.mean()
            else:
                raise ValueError(f"Unknown reduce: {self.reduce}")
        return scores  # [B], values in [0, 1]


class Q16VelocityScorer:
    """Differentiable Q16 (Schramowski et al. 2022) scorer.

    Mirrors eval_q16_official.Q16Official:
        ifeat       = CLIP_visual(image)
        sim         = (ifeat / |ifeat|) @ (P / |P|).T
        probs       = softmax(100.0 * sim)
        P(unsafe)   = probs[:, UNSAFE_IDX=0]      # row 0 = inappropriate in our prompts.p

    Wrapped as a callable(image[B,3,H,W] in [0,1]) → score[B] ∈ (0, 1) with full
    autograd through the CLIP vision tower so ScoreBasedGuidance{3,4}Z can take
    ∇_z log p(unsafe|z).
    """
    PROMPTS_PATH = "/workspace/Uzn/checkpoints/q16/prompts.p"
    UNSAFE_IDX = 0

    def __init__(self, device="cuda:0", model_name="ViT-L/14",
                 dtype=torch.float32):
        import clip, pickle
        self.device = torch.device(device)
        self.dtype = dtype
        model, _ = clip.load(model_name, device=self.device)
        self.model = model.to(dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        with open(self.PROMPTS_PATH, "rb") as f:
            arr = pickle.load(f)
        self.soft_prompts = torch.as_tensor(
            np.asarray(arr), dtype=dtype, device=self.device)
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
        ifeat = self.model.encode_image(x)
        ifeat = ifeat / (ifeat.norm(dim=-1, keepdim=True) + 1e-8)
        tfeat = self.soft_prompts / (
            self.soft_prompts.norm(dim=-1, keepdim=True) + 1e-8)
        logits = 100.0 * ifeat @ tfeat.T
        probs = logits.softmax(dim=-1)
        return probs[:, self.UNSAFE_IDX]


class LAIONNudityVelocityScorer:
    """Differentiable LAION CLIP-L NSFW classifier as in-loop safety scorer.

    Architecture (LAION-AI/CLIP-based-NSFW-Detector, autokeras export):
        image → CLIP-L/14 visual encoder → 768-d projection
              → Normalization (running mean/var)
              → Linear(768→64)  → ReLU
              → Linear(64→512)  → ReLU
              → Linear(512→256) → ReLU
              → Linear(256→1)   → sigmoid → P(NSFW) ∈ (0, 1)

    Independent training distribution from NudeNet (LAION-2B nsfw subset vs.
    NudeNet's manually-labeled set), so it can be used both as in-loop scorer
    and as an evaluator without metric/scorer cheating.
    """
    HEAD_PATH = "/workspace/Uzn/checkpoints/laion_nsfw/laion_nsfw_clipL.pt"

    def __init__(self, device="cuda:0", model_name="ViT-L/14",
                 dtype=torch.float32, head_path=None):
        import clip
        import torch.nn as nn
        self.device = torch.device(device)
        self.dtype = dtype
        model, _ = clip.load(model_name, device=self.device)
        self.clip_model = model.to(dtype=dtype).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        # Build head matching the LAION architecture
        self.head_path = head_path or self.HEAD_PATH
        sd = torch.load(self.head_path, map_location=self.device, weights_only=True)

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
        # CLIP normalization constants
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
        feats = self.clip_model.encode_image(x)
        feats = feats.to(self.dtype)
        # LAION head was trained on l2-normalized CLIP embeddings
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return self.head(feats)


class IterativeVSG:
    """
    Iterative Velocity Safety Guidance.

    Refines the velocity at a given denoising step by iteratively projecting out
    the unsafe component:
        v^{(0)} = v(x_t | c')
        x_0^{(k)} = x_t - sigma_t * v^{(k)}
        U_t = ∇_v scorer(x_0^{(k)}) / ||·||   (unit "unsafe" direction)
        v^{(k+1)} = v^{(k)} - λ_k · (U_t · v^{(k)}) · U_t

    Stops when `checker(x_0)` is safe, scorer score < `threshold`, or `max_iters`
    reached. Does NOT require transformer backward — only VAE+scorer.
    """

    def __init__(self, vae, scorer, threshold=0.2, max_iters=5,
                 lambda_base=1.0, lambda_decay=False, checker=None,
                 verbose=False, normalize_grad=True):
        self.vae = vae
        self.scorer = scorer
        self.threshold = threshold
        self.max_iters = max_iters
        self.lambda_base = lambda_base
        self.lambda_decay = lambda_decay
        self.checker = checker
        self.verbose = verbose
        self.normalize_grad = normalize_grad

    def _decode(self, x0_hat, scm_divide_sigma_data=False):
        """Decode packed FLUX latents to image in [0, 1].

        For sCM, x0_hat arrives in `sigma_data`-scale (matches pipeline line 1084)
        and must be divided by `sigma_data` before VAE decode (matches pipeline line 1116).
        """
        if scm_divide_sigma_data:
            x0_hat = x0_hat / self.sigma_data
        if x0_hat.dim() == 3:
            B, S, D = x0_hat.shape
            h = w = int(math.sqrt(S))
            c = D // 4
            x0_sp = x0_hat.view(B, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
            x0_sp = x0_sp.reshape(B, c, h * 2, w * 2)
        else:
            x0_sp = x0_hat
        vae_dev = next(self.vae.parameters()).device
        x0_scaled = (x0_sp / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x0_scaled = x0_scaled.to(vae_dev, dtype=self.vae.dtype)
        image = self.vae.decode(x0_scaled).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def correct(self, v_pred, x_t, sigma_t, denoised=None, scm_s=None, sigma_data=None):
        """
        Args:
            v_pred: [B, ...] velocity (or x_0 estimate for sCM) prediction
            x_t:    [B, ...] current latent
            sigma_t: scalar noise level (for flow-matching x_0 = x_t - σ·v path)
            denoised: [B, ...] optional pre-computed x_0 estimate (sCM trigflow path)
            scm_s:  float, raw sCM timestep (e.g. π/2, 1.3). When provided, use
                    trigflow formula:  x_0(v_g) = denoised + sin(s) · (v_ref - v_g)
            sigma_data: float, scheduler config sigma_data — needed for sCM scale conversion

        Returns:
            v_corrected (same shape/dtype as v_pred), history (list of scores)
        """
        v = v_pred.detach().clone()
        history = []
        use_scm = scm_s is not None and denoised is not None and sigma_data is not None
        if use_scm:
            sin_s = float(math.sin(scm_s))
            self.sigma_data = float(sigma_data)
            v_ref = v.detach().clone()
            denoised_ref = denoised.detach().clone()

        for k in range(self.max_iters):
            with torch.enable_grad():
                v_g = v.detach().clone().requires_grad_(True)
                if use_scm:
                    # trigflow: x_0 = denoised_ref + sin(s) · (v_ref - v_g)
                    # (equivalent to cos(s)·x_t - sin(s)·v_g; avoids needing x_t)
                    x0_hat = denoised_ref + sin_s * (v_ref - v_g)
                    image = self._decode(x0_hat, scm_divide_sigma_data=True)
                else:
                    x0_hat = x_t - sigma_t * v_g
                    image = self._decode(x0_hat)
                score = self.scorer(image)  # [B]

                if self.verbose and k == 0:
                    print(f"    [IterVSG-diag] grad_mode={torch.is_grad_enabled()} "
                          f"v_g.rg={v_g.requires_grad} "
                          f"x0_hat.gf={x0_hat.grad_fn is not None} "
                          f"image.rg={image.requires_grad} image.gf={image.grad_fn is not None} "
                          f"score.rg={score.requires_grad} score.gf={score.grad_fn is not None}")

                score_val = float(score.sum().item())

                # Stopping criterion (inside enable_grad so `image` keeps its grad graph live)
                stop = False
                if self.checker is not None:
                    is_safe = bool(self.checker(image.detach()))
                    if self.verbose:
                        print(f"    [IterVSG] iter {k}: score={score_val:.4f}, checker_safe={is_safe}")
                    stop = is_safe
                else:
                    if self.verbose:
                        print(f"    [IterVSG] iter {k}: score={score_val:.4f}, thr={self.threshold}")
                    stop = score_val < self.threshold

                history.append(score_val)
                if stop:
                    break

                # Gradient → unit direction (inside enable_grad to be safe)
                if not score.requires_grad:
                    # Cannot take grad; bail. Shouldn't happen, but guard.
                    print(f"    [IterVSG] WARN: score has no grad at iter {k}; bailing")
                    break
                grads = torch.autograd.grad(score.sum(), v_g)[0]

            grad_norm = grads.norm()
            if float(grad_norm) < 1e-8:
                break

            # Gradient descent on score.
            # normalize_grad=True:  v' = v − λ · ∇/‖∇‖     (unit-step; STG convention)
            # normalize_grad=False: v' = v − λ · ∇         (raw grad; scale ∝ score magnitude)
            lambda_k = self.lambda_base / (1.0 + k) if self.lambda_decay else self.lambda_base
            if self.normalize_grad:
                U_t = grads / (grad_norm + 1e-8)
                v = v - lambda_k * U_t
            else:
                v = v - lambda_k * grads

        return v.to(v_pred.dtype), history


# ============================================================================
# Main Interface
# ============================================================================

class SafeDenoiserForFLUX:
    """
    Safe Denoiser / SGF integration for FLUX-based pipelines.

    Provides callback_on_step_end that modifies latents during denoising
    to repel from unsafe reference images.
    """

    def __init__(self, vae=None, device="cuda:0"):
        """
        Args:
            vae: VAE model (for encoding unsafe images to latents). Not needed for uzn method.
            device: device for VAE encoding
        """
        self.vae = vae
        self.device = device
        self.ref_latents = None    # [N, C, H, W] for safe_denoiser/sgf

    def load_unsafe_images(self, path, image_size=512, max_images=None):
        """
        Load unsafe reference images and encode to latents.

        Args:
            path: directory of images or .pt file of pre-encoded latents
            image_size: resize images to this size
            max_images: limit number of images (None = all)
        """
        if path.endswith(".pt"):
            self.ref_latents = torch.load(path, map_location="cpu", weights_only=True)
            print(f"[SafeDenoiser] Loaded pre-encoded latents: {self.ref_latents.shape}")
            return

        # Load images from directory
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])

        images = []
        exts = (".png", ".jpg", ".jpeg", ".webp")
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(exts):
                img = Image.open(os.path.join(path, fname)).convert("RGB")
                images.append(transform(img))
                if max_images and len(images) >= max_images:
                    break

        if not images:
            raise ValueError(f"No images found in {path}")

        print(f"[SafeDenoiser] Encoding {len(images)} unsafe images...")
        images_tensor = torch.stack(images)

        # Encode in batches
        latents_list = []
        batch_size = 4
        vae_device = next(self.vae.parameters()).device
        with torch.no_grad():
            for i in range(0, len(images_tensor), batch_size):
                batch = images_tensor[i:i + batch_size].to(vae_device, dtype=self.vae.dtype)
                latent = self.vae.encode(batch).latent_dist.sample()
                latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                latents_list.append(latent.cpu())

        self.ref_latents = torch.cat(latents_list, dim=0)
        print(f"[SafeDenoiser] Encoded ref latents: {self.ref_latents.shape}")

    def save_ref_latents(self, path):
        """Save pre-encoded latents for faster loading."""
        if self.ref_latents is not None:
            torch.save(self.ref_latents, path)
            print(f"[SafeDenoiser] Saved ref latents to {path}")

    def get_callback(self, method="sgf", scale=0.03, warmup_start=0.8, warmup_end=0.4,
                     sigma=2.75, normalize=False, safree_iter_data=None):
        """
        Build callback_on_step_end for safe denoising.

        Matches paper implementation:
        - Safe Denoiser/SGF: predict x0_hat from noise_pred, apply repellency, renoise
        - Uzn: modify prompt_embeds directly (no x0_hat needed)

        Args:
            method: "safe_denoiser", "sgf", or "uzn"
            scale: repellency strength (paper: 0.03 for SDv3)
            warmup_start/end: fraction of steps (paper: t∈[780,1000] ≈ top 22%)
            sigma: kernel bandwidth (paper: 2.75 for SDv3, safe_denoiser only)
            normalize: normalize gradient (paper: False for SGF)
            safree_iter_data: dict from embedding_modifier for combined callback

        Returns:
            (callback_fn, callback_input_names)
        """
        # Create repellency method
        if method == "safe_denoiser":
            if self.ref_latents is None:
                raise ValueError("No reference latents loaded. Call load_unsafe_images() first.")
            repellency = SafeDenoiserRepellency(
                self.ref_latents, scale=scale, sigma=sigma)
        elif method == "sgf":
            if self.ref_latents is None:
                raise ValueError("No reference latents loaded. Call load_unsafe_images() first.")
            repellency = SGFRepellency(
                self.ref_latents, scale=scale, normalize=normalize)
        elif method == "vsg":
            if self.ref_latents is None:
                raise ValueError("No reference latents loaded. Call load_unsafe_images() first.")
            repellency = SGFRepellency(
                self.ref_latents, scale=scale, normalize=normalize)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'safe_denoiser', 'sgf', or 'vsg'.")

        n_ref = self.ref_latents.shape[0]
        print(f"[SafeDenoiser] method={method}, scale={scale}, sigma={sigma}, "
              f"normalize={normalize}, warmup=[{warmup_start}, {warmup_end}], refs={n_ref}")

        _warmup_start = warmup_start
        _warmup_end = warmup_end
        _safree_data = safree_iter_data
        _is_vsg = (method == "vsg")

        def _unpack_flux(latents):
            """FLUX packed [B, S, C*4] → unpacked [B, C, H, W]."""
            B, S, D = latents.shape
            h = w = int(math.sqrt(S))
            c = D // 4
            x = latents.view(B, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
            return x.reshape(B, c, h * 2, w * 2), (h, w, c)

        def _repack_flux(x, h, w, c):
            """Unpacked [B, C, H, W] → FLUX packed [B, S, C*4]."""
            B = x.shape[0]
            x = x.reshape(B, c, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5)
            return x.reshape(B, h * w, c * 4)

        def _apply_repellency_to_x0(x0_hat):
            """Apply repellency to unpacked x0_hat [B, C, H, W]."""
            ref_h, ref_w = repellency.ref_latents.shape[2], repellency.ref_latents.shape[3]
            if x0_hat.shape[2] != ref_h or x0_hat.shape[3] != ref_w:
                resized_ref = F.interpolate(
                    repellency.ref_latents.to(x0_hat.device, dtype=x0_hat.dtype),
                    size=(x0_hat.shape[2], x0_hat.shape[3]), mode="bilinear", align_corners=False)
                orig_ref = repellency.ref_latents
                repellency.ref_latents = resized_ref
                result = repellency.compute(x0_hat)
                repellency.ref_latents = orig_ref
                return result
            return repellency.compute(x0_hat)

        def _callback(pipeline, step_idx, timestep, callback_kwargs):
            total_steps = pipeline._num_timesteps
            progress = step_idx / max(total_steps, 1)  # 0 → 1

            apply_repellency = (progress <= (1.0 - _warmup_end)) and (progress >= (1.0 - _warmup_start))

            if apply_repellency:
                # --- Latent-space methods: need noise_pred for x0 prediction ---
                latents = callback_kwargs["latents"]
                noise_pred = callback_kwargs.get("noise_pred")

                if noise_pred is None:
                    pass
                else:
                    lat_shape = latents.shape
                    is_packed = len(lat_shape) == 3

                    # Get sigma from scheduler
                    scheduler = pipeline.scheduler
                    idx = step_idx
                    if hasattr(scheduler, 'sigmas'):
                        sigmas = scheduler.sigmas
                        sigma_cur = float(sigmas[idx])
                        sigma_next = float(sigmas[idx + 1]) if idx + 1 < len(sigmas) else 0.0
                    else:
                        t_val = float(timestep)
                        sigma_cur = t_val / 1000.0
                        sigma_next = max(sigma_cur - 1.0 / total_steps, 0.0)

                    # Unpack if FLUX
                    if is_packed:
                        x_post, pack_info = _unpack_flux(latents)
                        v_pred, _ = _unpack_flux(noise_pred)
                    else:
                        x_post = latents
                        v_pred = noise_pred

                    # Reconstruct pre-step x_t
                    x_t_pre = x_post + (sigma_cur - sigma_next) * v_pred
                    x0_hat = x_t_pre - sigma_cur * v_pred

                    # Apply repellency to x0_hat
                    x0_safe = _apply_repellency_to_x0(x0_hat)

                    if _is_vsg:
                        # --- VSG: modify velocity, standard step (no renoise) ---
                        # delta_v = x0_hat - x0_safe (unsafe direction in x0 space)
                        # v' = v + delta_v → x0' moves toward x0_safe
                        # Effect: Δx0 = -σ_t * delta_v (natural σ_t weighting)
                        delta_v = x0_hat - x0_safe
                        v_new = v_pred + delta_v
                        # Standard Euler step with modified velocity
                        dt = sigma_next - sigma_cur
                        latents_new = x_t_pre + dt * v_new
                    else:
                        # --- SGF / Safe Denoiser: x0 modification + renoise ---
                        x1_hat = x_t_pre + (1.0 - sigma_cur) * v_pred
                        delta = sigma_cur - sigma_next
                        noise = (math.sqrt(sigma_next) * x1_hat
                                 + math.sqrt(max(1.0 - sigma_next, 0.0)) * torch.randn_like(x1_hat))
                        latents_new = x0_safe + (sigma_cur - delta) * (noise - x0_safe)

                    if is_packed:
                        h, w, c = pack_info
                        callback_kwargs["latents"] = _repack_flux(latents_new, h, w, c).to(latents.dtype)
                    else:
                        callback_kwargs["latents"] = latents_new.to(latents.dtype)

            # --- SAFREE iterative projection (if combined) ---
            if _safree_data is not None:
                pe = callback_kwargs.get("prompt_embeds")
                if pe is not None:
                    text_e = pe.squeeze(0).float()
                    _V = _safree_data["V_r"].to(text_e.device, dtype=text_e.dtype)
                    _I = _safree_data["I_m_P_r"].to(text_e.device, dtype=text_e.dtype)
                    _m = _safree_data["attn_mask"].to(text_e.device)
                    t_low = text_e @ _V
                    t_safe = (_I @ t_low.T).T
                    delta = _safree_data["strength"] * (t_safe - t_low)
                    t_proj = text_e + (delta @ _V.T)
                    mask = _m.unsqueeze(1)
                    merged = text_e * (1.0 - mask) + t_proj * mask
                    callback_kwargs["prompt_embeds"] = merged.unsqueeze(0).to(pe.dtype)

            return callback_kwargs

        cb_inputs = ["latents", "noise_pred"]
        if _safree_data is not None:
            cb_inputs.append("prompt_embeds")

        return _callback, cb_inputs




