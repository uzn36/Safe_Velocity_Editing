"""
SGF and STG — comparison safety methods adapted for MeanFlow (FLUX).

SGF (ICLR 2026 Oral, `kaistAI/SGF`)
  ~ MMD-gradient repellency on latents from unsafe reference images.
  ~ At each step: recover x0_hat from (latents, noise_pred, sigma), push x0_hat
    away from ref-latent cluster via -∇_x K(x, refs), renoise back to x_{t+1}.
  ~ Paper uses DDPM; here we re-noise with the flow-matching relation
    x_{t'} = (1 - σ') · x0 + σ' · x1_hat.

STG (NeurIPS 2025, `aailab-kaist/STG`)
  ~ Gradient of a NudeNet safety score w.r.t. text embeddings.
  ~ At chosen steps: run transformer w/ grad on prompt_embeds, predict x0_hat,
    VAE-decode, score with NudeNet, backprop and update prompt_embeds by
    `prompt_embeds -= lr · grad / ||grad||`.

Both plug into `FluxPipeline_MeanFlow.__call__` via `callback_on_step_end`
using the `noise_pred` tensor exposed by `MultiGPUFluxPipeline`.
"""

import math
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ============================================================================
# FLUX packing / unpacking (matches MeanFlow / Flux tokenization of latents)
# ============================================================================

def _unpack_flux(latents):
    """Packed [B, S, C*4] → unpacked [B, C, H, W] where H=W=2*sqrt(S)."""
    B, S, D = latents.shape
    h = w = int(math.sqrt(S))
    c = D // 4
    x = latents.view(B, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, c, h * 2, w * 2), (h, w, c)


def _repack_flux(x, h, w, c):
    """Unpacked [B, C, 2h, 2w] → packed [B, h*w, C*4]."""
    B = x.shape[0]
    x = x.reshape(B, c, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, h * w, c * 4)


def _get_sigmas(scheduler, step_idx, timestep, total_steps):
    """Return (sigma_cur, sigma_next) for a FlowMatchEulerDiscreteScheduler step."""
    if hasattr(scheduler, "sigmas"):
        sigmas = scheduler.sigmas
        s_cur = float(sigmas[step_idx])
        s_next = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
    else:
        s_cur = float(timestep) / 1000.0
        s_next = max(s_cur - 1.0 / total_steps, 0.0)
    return s_cur, s_next


# ============================================================================
# SGF — MMD gradient repellency on x0_hat (latent space)
# ============================================================================

class SGFMeanFlow:
    """
    SGF adapted for MeanFlow / FLUX.

    Reference: `repellency_methods_sgf.py` (grad_mmd) from the official repo.
    The original does: dK/dx = Σ_j{-2γ K(x,y_j) (x - y_j)} with adaptive γ from
    top-k squared distances, then applies `x0_hat += scale · dK_dX` with scale<0
    (i.e. repel). This port expresses it as `x0_safe = x0_hat - scale · dK_dX`
    (scale > 0) so that scale reads as an intensity.
    """

    def __init__(self, vae, vae_device="cuda:2", transformer_device="cuda:1"):
        self.vae = vae
        self.vae_device = torch.device(vae_device)
        self.transformer_device = torch.device(transformer_device)
        self.ref_latents = None  # [N, C, H, W], lives on transformer device for callback use

    # ---------- reference image loading ---------------------------------------
    def load_unsafe_images(self, path, image_size=512, max_images=None):
        """Encode reference unsafe images to FLUX VAE latents."""
        if path.endswith(".pt"):
            self.ref_latents = torch.load(path, map_location="cpu", weights_only=True)
            print(f"[SGF] Loaded pre-encoded latents: {self.ref_latents.shape}")
            self.ref_latents = self.ref_latents.to(self.transformer_device)
            return

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        exts = (".png", ".jpg", ".jpeg", ".webp")
        images = []
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(exts):
                img = Image.open(os.path.join(path, fname)).convert("RGB")
                images.append(transform(img))
                if max_images and len(images) >= max_images:
                    break
        if not images:
            raise ValueError(f"[SGF] No images found under {path}")

        images_tensor = torch.stack(images)
        latents_list = []
        with torch.no_grad():
            for i in range(0, len(images_tensor), 4):
                batch = images_tensor[i:i + 4].to(self.vae_device, dtype=self.vae.dtype)
                lat = self.vae.encode(batch).latent_dist.sample()
                lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                latents_list.append(lat.cpu())
        self.ref_latents = torch.cat(latents_list, dim=0).to(self.transformer_device)
        print(f"[SGF] Encoded {self.ref_latents.shape[0]} ref images → "
              f"latents {tuple(self.ref_latents.shape)}")

    def save_ref_latents(self, path):
        if self.ref_latents is not None:
            torch.save(self.ref_latents.cpu(), path)

    # ---------- MMD gradient (adaptive RBF bandwidth) -------------------------
    @staticmethod
    def _adaptive_gamma(X_flat, Y_flat, k=3, eps=0.05, eps_div=1e-8):
        """γ = -log(eps) / mean(top-k squared distances) — matches SGF original."""
        dist_sq = torch.cdist(X_flat, Y_flat).pow(2)
        topk_sq, _ = dist_sq.topk(min(k, dist_sq.shape[1]), dim=1, largest=False)
        r_k2 = topk_sq.mean()
        return -math.log(eps) / (r_k2 + eps_div)

    def _grad_mmd(self, x0_hat, refs, normalize=False):
        """Return dK/dx ∈ R^{B×C×H×W} — gradient of RBF kernel sum w.r.t. x."""
        B, C, H, W = x0_hat.shape
        x_flat = x0_hat.reshape(B, -1)
        ref_flat = refs.reshape(refs.shape[0], -1)
        gamma = self._adaptive_gamma(x_flat, ref_flat)
        dist_sq = torch.cdist(x_flat, ref_flat).pow(2)
        K = (-gamma * dist_sq).exp()
        diff = x_flat.unsqueeze(1) - ref_flat.unsqueeze(0)        # [B, N, D]
        dK = (-2.0 * gamma * K.unsqueeze(-1) * diff).sum(dim=1)   # [B, D]
        if normalize:
            nrm = dK.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            dK = dK / nrm
        return dK.reshape(B, C, H, W)

    def _resize_refs(self, target_shape):
        """Bilinearly resize reference latents to match the runtime spatial size."""
        H, W = target_shape[-2], target_shape[-1]
        if self.ref_latents.shape[-2:] == (H, W):
            return self.ref_latents
        return F.interpolate(self.ref_latents.float(),
                             size=(H, W), mode="bilinear", align_corners=False)

    # ---------- main callback -------------------------------------------------
    def get_callback(self, scale=0.015, warmup_start=0.8, warmup_end=0.4,
                     normalize=False, safree_iter_data=None,
                     stochastic_renoise=True, sign="paper",
                     generator=None, noise_mode="nudity"):
        """
        Returns (callback_fn, callback_input_names) for FluxPipeline_MeanFlow.

        Reproduces `modified_safree_diffusion_pipeline_sgf.py:546-568`:

        ```python
        # (1)  t ∈ [neg_end, neg_start]  warmup gate
        # (2)  x_0_hat = scheduler.step(noise_pred, t, latents).pred_original_sample
        # (3)  x_0_hat_repelled = repel(x_0_hat)          # grad_mmd
        # (4)  noise = randn(...)                         # FRESH noise
        # (5)  latents = scheduler.add_noise(x_0_hat_repelled, noise, t)
        # (6)  latents = scheduler.step(noise_pred, t, latents).prev_sample
        ```

        For MeanFlow / flow-matching this maps to:
            x_t_new = (1 − σ_cur)·x0_safe  +  σ_cur · {ε_fresh  or  x1_hat}
            x_{t-1} = x_t_new + (σ_next − σ_cur) · v_pred_OLD

        Args:
            stochastic_renoise: True → fresh Gaussian noise (paper-faithful).
                                False → reuse x1_hat (deterministic).
            sign: "paper" → `x_0_hat += scale · dK_dX`  (literal repellency_methods_sgf.py)
                  "intent" → `x_0_hat -= scale · dK_dX`  (standard repel-from-refs)
        """
        if self.ref_latents is None:
            raise ValueError("[SGF] Call load_unsafe_images(...) before get_callback.")

        if sign not in ("paper", "intent"):
            raise ValueError(f"[SGF] sign must be 'paper' or 'intent', got {sign!r}")
        _sign_mul = +1.0 if sign == "paper" else -1.0

        _warmup_start, _warmup_end = warmup_start, warmup_end
        _scale = scale
        _normalize = normalize
        _stochastic = stochastic_renoise
        _safree_data = safree_iter_data
        print(f"[SGF] scale={scale}, warmup=[{warmup_start},{warmup_end}], "
              f"stochastic_renoise={stochastic_renoise}, sign={sign}")

        def _cb(pipeline, step_idx, timestep, callback_kwargs):
            total = pipeline._num_timesteps
            progress = step_idx / max(total, 1)
            in_window = (progress <= (1.0 - _warmup_end)) \
                and (progress >= (1.0 - _warmup_start))

            if in_window:
                latents = callback_kwargs["latents"]
                noise_pred = callback_kwargs.get("noise_pred")
                if noise_pred is not None:
                    s_cur, s_next = _get_sigmas(pipeline.scheduler, step_idx,
                                                timestep, total)

                    packed = latents.dim() == 3
                    if packed:
                        x_post, pack = _unpack_flux(latents)
                        v_pred, _ = _unpack_flux(noise_pred)
                    else:
                        x_post, v_pred = latents, noise_pred

                    # Flow-matching identities (x_t = (1-σ)·x0 + σ·x1, v = x1 - x0)
                    x_t_pre = x_post + (s_cur - s_next) * v_pred
                    x0_hat = x_t_pre - s_cur * v_pred
                    x1_hat = x_t_pre + (1.0 - s_cur) * v_pred

                    refs = self._resize_refs(x0_hat.shape).to(
                        x0_hat.device, dtype=x0_hat.dtype)
                    dK = self._grad_mmd(x0_hat.float(), refs.float(),
                                        normalize=_normalize).to(x0_hat.dtype)
                    # ─── sign convention ───
                    # paper  : x0_safe = x0_hat + scale · dK   (literal code)
                    # intent : x0_safe = x0_hat − scale · dK   (repel from refs)
                    x0_safe = x0_hat + _sign_mul * _scale * dK

                    # ─── Hybrid re-compose (paper-faithful) ────────────────
                    # Matches SGF/diversity_sdv3/models/sdv3/safer_pipeline.py:
                    #   noise   = √σ_next · x_1_hat + √(1-σ_next) · ε_fresh
                    #   latents = (1-σ_next) · x_0_safe + σ_next · noise
                    # σ_next → 0  ⇒  latents → x_0_safe   (clean)
                    # σ_next → 1  ⇒  latents → x_1_hat    (pure noise)
                    #
                    # Keeping x_1_hat (not replacing with fresh ε) preserves
                    # the original trajectory's noise-endpoint prediction,
                    # which is essential for paper-faithful behaviour: our
                    # previous "simple" formula (σ_next·ε only) threw x_1_hat
                    # away and produced noticeably STRONGER safety than paper.
                    # noise_mode selects the re-noise recipe (see sgf_sd3.py
                    # for the full explanation).  "nudity" (default) matches
                    # SDv1.4 SGF and is what our original MeanFlow port was
                    # empirically validated against; "diversity" matches the
                    # official SDv3 SGF (hybrid noise preserving x_1).
                    if _stochastic:
                        eps = torch.randn(
                            x0_safe.shape, device=x0_safe.device,
                            dtype=x0_safe.dtype, generator=generator,
                        )
                    else:
                        eps = None

                    if noise_mode == "nudity":
                        noise_term = eps if _stochastic else x1_hat
                    elif noise_mode == "diversity":
                        if _stochastic:
                            noise_term = (math.sqrt(max(s_next, 0.0)) * x1_hat
                                          + math.sqrt(max(1.0 - s_next, 0.0)) * eps)
                        else:
                            noise_term = x1_hat
                    else:
                        raise ValueError(
                            f"[SGF] noise_mode must be 'nudity' or "
                            f"'diversity', got {noise_mode!r}")
                    latents_new = (1.0 - s_next) * x0_safe + s_next * noise_term

                    if packed:
                        h, w, c = pack
                        callback_kwargs["latents"] = \
                            _repack_flux(latents_new, h, w, c).to(latents.dtype)
                    else:
                        callback_kwargs["latents"] = latents_new.to(latents.dtype)

            # Optional: SAFREE iterative embedding projection on top of SGF.
            if _safree_data is not None and "prompt_embeds" in callback_kwargs:
                pe = callback_kwargs["prompt_embeds"]
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

        inputs = ["latents", "noise_pred"]
        if _safree_data is not None:
            inputs.append("prompt_embeds")
        return _cb, inputs


# ============================================================================
# STG — gradient of a NudeNet safety score w.r.t. prompt_embeds
# ============================================================================

class _NudeNetYOLO(nn.Module):
    """
    Thin wrapper around a YOLOv8 NudeNet checkpoint (from `STG/nudenet`).

    `safety_score(img_tensor)` returns (scores, masks) where `scores[i]` is a
    differentiable sum of detection confidences for the unsafe classes in
    sample i, and `masks[i]` is the number of positive detections.
    """

    # Default unsafe class ids in the 640m.pt label set
    UNSAFE_CLASS_IDS = [3, 4, 6, 14]  # breast/genitalia/anus/male_genitalia

    def __init__(self, checkpoint_path, device, conf=0.01,
                 inference_resolution=640, unsafe_class_ids=None):
        super().__init__()
        from ultralytics import YOLO  # lazy import
        try:
            from ultralytics.utils.nms import non_max_suppression  # ultralytics ≥ 8.4
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression  # older fallback
        self._nms = non_max_suppression
        self.model = YOLO(checkpoint_path)
        self.model.to(device)
        self.device = torch.device(device)
        self.input_hw = (inference_resolution, inference_resolution)
        self.resize = transforms.Resize(self.input_hw)
        self.predictor = self.model._smart_load("predictor")(
            overrides={"conf": conf, "batch": 1, "save": False, "mode": "predict"}
        )
        self.predictor.setup_model(model=self.model.model)
        self.unsafe_ids = torch.tensor(
            unsafe_class_ids or self.UNSAFE_CLASS_IDS, device=self.device)

    @property
    def _args(self):
        return self.predictor.args

    def safety_score(self, img_tensor):
        """
        img_tensor: [B, 3, H, W] in [0, 1]. Returns (scores[B], masks[B]).
        """
        x = self.resize(img_tensor).clamp(0, 1)
        if x.device != self.device:
            x = x.to(self.device)
        # YOLO weights are float32; upcast to match (keeps autograd graph alive).
        weight_dtype = next(self.model.model.parameters()).dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        out = self.model.model(x)
        preds = self._nms(
            out, self._args.conf, self._args.iou,
            self._args.classes, self._args.agnostic_nms,
            max_det=self._args.max_det, nc=len(self.model.names),
            end2end=getattr(self._args.model, "end2end", False),
            rotated=self._args.task == "obb",
        )
        scores = torch.zeros((len(preds),), device=x.device)
        masks = torch.zeros((len(preds),), device=x.device)
        for i, pred in enumerate(preds):
            if pred.numel() == 0:
                continue
            m = torch.isin(pred[:, 5], self.unsafe_ids)
            scores[i] = pred[m, 4].sum()
            masks[i] = m.sum()
        return scores, masks


class STGMeanFlow:
    """
    STG adapted for MeanFlow / FLUX.

    Core loop (at each step where `update_fn(i) == True`):
      1. Reconstruct x_t_pre from (latents_post, noise_pred).
      2. Re-run transformer with grad on prompt_embeds → v_pred_grad.
      3. Recover x0_hat, unpack, scale, VAE-decode → image.
      4. r = NudeNet.safety_score(image); L = r + λ · ||Δe||².
      5. g = ∂L/∂prompt_embeds; update prompt_embeds -= lr · g / ||g||.

    Matches `pipeline_stable_diffusion_stg.py` but targets the FLUX MeanFlow
    transformer signature (dual timestep + guidance embedding + image/text ids).
    """

    def __init__(
        self,
        transformer,
        vae,
        transformer_device="cuda:1",
        vae_device="cuda:2",
        nudenet_path=None,
        nudenet_device=None,
        conf=0.01,
        scorer=None,
    ):
        """Mirror of `stg_flux.STGForFlux.__init__`: accepts either a NudeNet
        path (default — nudity erasure) or a generic `scorer(image)` callable
        returning rewards (e.g. CLIPVelocityScorer for violence/style erasure).
        """
        self.transformer = transformer
        self.vae = vae
        self.transformer_device = torch.device(transformer_device)
        self.vae_device = torch.device(vae_device)
        if scorer is not None:
            self.scorer = scorer
            self._is_nudenet = hasattr(scorer, "safety_score")
        elif nudenet_path is not None:
            nudenet_device = nudenet_device or transformer_device
            self.scorer = _NudeNetYOLO(
                nudenet_path, device=nudenet_device, conf=conf)
            self._is_nudenet = True
        else:
            raise ValueError(
                "[STG] Either `scorer` or `nudenet_path` is required.")
        self.nude_net = self.scorer if self._is_nudenet else None

        # Static conditioning (filled by `prime(...)`).
        self.latent_image_ids = None
        self.text_ids = None
        self.pooled_prompt_embeds = None
        self.guidance = None
        self.prompt_embeds_given = None  # for L2 prior

    def prime(self, *, latent_image_ids, text_ids, pooled_prompt_embeds,
              guidance, prompt_embeds_given):
        """Store step-invariant conditioning needed for the STG transformer re-forward."""
        self.latent_image_ids = latent_image_ids
        self.text_ids = text_ids
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.guidance = guidance
        self.prompt_embeds_given = prompt_embeds_given.detach().clone()

    # ---------- main callback -------------------------------------------------
    def get_callback(
        self,
        lr=0.1,
        update_itrs=None,       # explicit list of step indices
        update_intervals=None,  # (start, end) range
        update_freq=1,          # every-k fallback
        weight_prior=0.0,       # L2 regularizer toward original embeds
        init_org=False,         # reset to original embeds at each update
    ):
        # Priming is checked lazily when the callback fires, because the
        # caller typically builds the callback before pipeline call-time
        # inputs (latent_image_ids etc.) are known.
        if update_itrs is not None:
            _should_update = lambda i: i in set(update_itrs)
        elif update_intervals is not None:
            lo, hi = update_intervals
            _should_update = lambda i: lo <= i < hi
        else:
            _should_update = lambda i: (i % update_freq == 0)

        def _transformer_forward(x_t, t_vec, r_t, encoder_hidden_states):
            """Single MeanFlow transformer forward using primed static conditioning."""
            return self.transformer(
                hidden_states=x_t,
                timestep=t_vec.flatten() / 1000.0,
                timestep2=(t_vec - r_t).flatten() / 1000.0,
                guidance=self.guidance,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=self.text_ids,
                img_ids=self.latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        def _cb(pipeline, step_idx, timestep, callback_kwargs):
            if not _should_update(step_idx):
                return callback_kwargs
            if self.latent_image_ids is None:
                raise RuntimeError(
                    "[STG] prime(...) must be called with static conditioning "
                    "before the first callback fires.")

            latents_post = callback_kwargs["latents"]          # x_{t-1}
            noise_pred_old = callback_kwargs.get("noise_pred")  # v_pred used for this step
            prompt_embeds = callback_kwargs.get("prompt_embeds")
            if noise_pred_old is None or prompt_embeds is None:
                return callback_kwargs

            total = pipeline._num_timesteps
            s_cur, s_next = _get_sigmas(pipeline.scheduler, step_idx, timestep, total)

            # Reconstruct x_t (pre-step latents).  Euler reverse:
            #   x_post = x_pre + (σ_next − σ_cur) · v_pred  ⇒
            #   x_pre  = x_post + (σ_cur − σ_next) · v_pred
            x_t = latents_post + (s_cur - s_next) * noise_pred_old

            # Dual-timestep inputs (MeanFlow signature).
            timesteps_all = pipeline.scheduler.timesteps
            t_vec = timestep.expand(x_t.shape[0]).to(x_t.dtype) \
                if torch.is_tensor(timestep) else torch.full(
                    (x_t.shape[0],), float(timestep), device=x_t.device,
                    dtype=x_t.dtype)
            if step_idx + 1 < len(timesteps_all):
                r_t = timesteps_all[step_idx + 1].expand(x_t.shape[0]).to(x_t.dtype)
            else:
                r_t = torch.zeros_like(t_vec)

            # ═════════════════════ (1) GRAD PASS ══════════════════════════
            # Faithful to pipeline_stable_diffusion_stg.py:1066-1097:
            #   - requires_grad on text embeddings
            #   - unet forward → noise_pred (w/ CFG in original; single forward
            #     here because MeanFlow internalises CFG into guidance embedding)
            #   - scheduler.step → pred_original_sample (x0_hat)
            #   - VAE decode → NudeNet safety_score → backprop
            pe_grad = prompt_embeds.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                v_grad = _transformer_forward(x_t, t_vec, r_t, pe_grad)

                # Flow-matching x_0 prediction  (≡ scheduler.pred_original_sample)
                x0_hat_packed = x_t - s_cur * v_grad
                x0_hat, _ = _unpack_flux(x0_hat_packed)

                x0_vae = x0_hat.to(self.vae_device, dtype=self.vae.dtype)
                x0_vae = (x0_vae / self.vae.config.scaling_factor) \
                    + self.vae.config.shift_factor
                image = self.vae.decode(x0_vae, return_dict=False)[0]
                image = (image / 2 + 0.5)

                if self._is_nudenet:
                    rewards, masks = self.scorer.safety_score(image)
                    rewards = rewards.to(pe_grad.device)
                    masks = masks.to(pe_grad.device)
                else:
                    rewards = self.scorer(image)
                    if not torch.is_tensor(rewards):
                        rewards = torch.tensor(rewards, device=pe_grad.device)
                    rewards = rewards.to(pe_grad.device)
                    if rewards.dim() == 0:
                        rewards = rewards.unsqueeze(0)
                    masks = torch.ones_like(rewards)
                loss = rewards.sum()
                if weight_prior > 0.0:
                    pe_given = self.prompt_embeds_given.to(pe_grad.device,
                                                           dtype=pe_grad.dtype)
                    prior = ((pe_given - pe_grad) ** 2).mean()
                    loss = loss + weight_prior * prior

                if masks.sum() <= 0:
                    # No unsafe detected → skip both update AND redo
                    return callback_kwargs

                grad = torch.autograd.grad(loss, pe_grad, retain_graph=False)[0]

            # ═════════════════════ (2) EMBED UPDATE ═══════════════════════
            # Faithful to pipeline_stable_diffusion_stg.py:1103-1111:
            #   grads_norm = grad / ||grad||
            #   prompt_embeds -= lr · grads_norm
            #   (or = pe_given − lr · grads_norm if init_org)
            with torch.no_grad():
                g_flat = grad.reshape(grad.shape[0], -1)
                nrm = g_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
                grad_norm = (g_flat / nrm).reshape_as(grad)

                if init_org:
                    pe_updated = self.prompt_embeds_given.to(
                        prompt_embeds.device, dtype=prompt_embeds.dtype
                    ) - lr * grad_norm
                else:
                    pe_updated = prompt_embeds - lr * grad_norm

            # ═════════════════════ (3) STEP REDO ══════════════════════════
            # Faithful to pipeline_stable_diffusion_stg.py:1113-1126:
            #   - Re-run unet with UPDATED prompt embeddings (no grad)
            #   - CFG combine
            #   - scheduler.step → latents_next_update
            #   - replace latents_[masks] = latents_next_update
            with torch.no_grad():
                v_new = _transformer_forward(x_t, t_vec, r_t, pe_updated)
                # Flow-matching Euler step: x_{t-1} = x_t + (σ_next − σ_cur) · v
                latents_new = x_t + (s_next - s_cur) * v_new

            callback_kwargs["latents"] = latents_new.to(latents_post.dtype)
            callback_kwargs["prompt_embeds"] = pe_updated.to(prompt_embeds.dtype)

            # Clear graph-related state to avoid memory leaks
            del pe_grad, v_grad, x0_hat_packed, x0_hat, x0_vae, image, grad
            torch.cuda.empty_cache()

            return callback_kwargs

        return _cb, ["latents", "prompt_embeds", "noise_pred"]


# ============================================================================
# latent_image_ids helper (static, pipeline-agnostic)
# ============================================================================

def prepare_latent_image_ids(pipeline, batch_size, height, width, device, dtype):
    """
    Delegate to the pipeline's own `_prepare_latent_image_ids` so shape/dtype
    exactly match what the FLUX transformer expects (handles batch repeat +
    reshape internally).
    """
    h = 2 * (int(height) // pipeline.vae_scale_factor)
    w = 2 * (int(width) // pipeline.vae_scale_factor)
    return pipeline._prepare_latent_image_ids(batch_size, h, w, device, dtype)
