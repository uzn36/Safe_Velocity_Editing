#!/usr/bin/env python3
"""
MeanFlow with SAFREE - Text-to-Image Generation with Concept Erasure
Based on T2I-Distill (Alibaba DAMO) + Official SAFREE

SAFREE Official: https://github.com/jaehong31/SAFREE

Usage:
    # Basic MeanFlow
    python sample_meanflow_safree.py --prompt "a cat sitting on grass"

    # MeanFlow + SAFREE
    python sample_meanflow_safree.py --prompt "a naked woman" --safree --unsafe-concepts nudity naked nude
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

# Add T2I-Distill to path
sys.path.insert(0, "/workspace/T2I-Distill")

from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from model.mmdit_flux import FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1
from model.pipeline import FluxPipeline_MeanFlow
from embedding_modifier import *


# ============================================================================
# Multi-GPU Pipeline Wrapper
# ============================================================================

class MultiGPUFluxPipeline(FluxPipeline_MeanFlow):
    """
    Subclass of FluxPipeline_MeanFlow that properly handles multi-GPU device placement.

    The issue: Base pipeline uses _execution_device property which returns wrong device
    in multi-GPU setups, causing latent_image_ids to be created on wrong GPU.

    Fix: Override _execution_device to return transformer device (main compute device).
    """

    # Add noise_pred to allowed callback inputs (needed for x0_hat-based repellency)
    _callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred"]

    _transformer_device_override = None  # Class-level default

    def set_transformer_device(self, device):
        """Set the transformer device for multi-GPU support."""
        self._transformer_device_override = device

    @property
    def _execution_device(self):
        """Override to return transformer device for proper multi-GPU support."""
        if self._transformer_device_override is not None:
            return torch.device(self._transformer_device_override)
        # Fallback to transformer's actual device
        if hasattr(self, 'transformer') and self.transformer is not None:
            return next(self.transformer.parameters()).device
        return super()._execution_device

# ============================================================================
# MeanFlow Pipeline with SAFREE
# ============================================================================

class MeanFlowSAFREEPipeline:
    """MeanFlow pipeline with optional SAFREE concept erasure and Multi-GPU support."""

    def __init__(
        self,
        pretrained_model_path="Freepik/flux.1-lite-8B",
        meanflow_checkpoint_path=None,
        # Multi-GPU 설정
        text_encoder_device="cuda:0",   # CLIP + T5 (~11GB)
        transformer_device="cuda:1",     # Transformer (~16GB)
        vae_device="cuda:2",             # VAE (~1GB)
        dtype=torch.bfloat16,
    ):
        self.text_encoder_device = text_encoder_device
        self.transformer_device = transformer_device
        self.vae_device = vae_device
        self.dtype = dtype

        # 메인 device는 transformer device로 설정 (sampling용)
        self.device = transformer_device

        print("=" * 70)
        print("Loading MeanFlow Pipeline (Multi-GPU)")
        print("=" * 70)
        print(f"  Text Encoders: {text_encoder_device}")
        print(f"  Transformer:   {transformer_device}")
        print(f"  VAE:           {vae_device}")
        print("=" * 70)

        # Load scheduler (CPU, no GPU needed)
        print("[1/5] Loading scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        )

        # Load tokenizers (CPU, no GPU needed)
        print("[2/5] Loading tokenizers...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2"
        )

        # Load text encoders → GPU 0
        print(f"[3/5] Loading text encoders → {text_encoder_device}...")

        # Load CLIP directly to target device
        print(f"  Loading CLIP text encoder...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).to(text_encoder_device)
        self.text_encoder.requires_grad_(False)
        print(f"  ✓ CLIP: {text_encoder_device}")

        # Load T5 directly to target device
        print(f"  Loading T5-XXL text encoder...")
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            pretrained_model_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).to(text_encoder_device)
        self.text_encoder_2.requires_grad_(False)
        print(f"  ✓ T5-XXL: {text_encoder_device}")

        # Load VAE → GPU 2
        print(f"[4/5] Loading VAE → {vae_device}...")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path,
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).to(vae_device)
        self.vae.requires_grad_(False)
        print(f"  ✓ VAE: {vae_device}")

        # Load transformer → GPU 1 (가장 큰 모델)
        print(f"[5/5] Loading MeanFlow transformer → {transformer_device}...")

        # Load MeanFlow checkpoint if provided
        if meanflow_checkpoint_path:
            print(f"  Loading MeanFlow weights from: {meanflow_checkpoint_path}")

            # Method 1: Load base model without meta tensors, then load checkpoint
            self.transformer = FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1.from_pretrained(
                pretrained_model_path,
                subfolder="transformer",
                low_cpu_mem_usage=False,  # Avoid meta tensors
                torch_dtype=dtype,
            )

            # Load MeanFlow checkpoint with assign=True for proper weight assignment
            state_dict = torch.load(meanflow_checkpoint_path, map_location='cpu', weights_only=False)
            missing, unexpected = self.transformer.load_state_dict(state_dict["module"], strict=False, assign=True)
            if missing:
                print(f"  Warning: Missing keys: {len(missing)} keys")
            if unexpected:
                print(f"  Warning: Unexpected keys: {len(unexpected)} keys")

            # Now move to target device
            self.transformer = self.transformer.to(transformer_device)
        else:
            # No checkpoint, just load base model directly to device
            self.transformer = FluxTransformer2DModelDualTimestepTimestepX1GuidanceX1.from_pretrained(
                pretrained_model_path,
                subfolder="transformer",
                torch_dtype=dtype,
                device_map=transformer_device,
            )

        self.transformer.requires_grad_(False)
        self.transformer.eval()
        print(f"  ✓ Transformer: {transformer_device}")

        # Create pipeline with multi-GPU support
        # Use our custom subclass that properly handles _execution_device
        self.pipeline = MultiGPUFluxPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
        )
        # Set transformer device for _execution_device override
        self.pipeline.set_transformer_device(transformer_device)

        # Store transformer device for multi-GPU execution
        self._transformer_device = transformer_device

        # GPU 메모리 사용량 출력
        self._print_gpu_memory()

        print("=" * 70)
        print("Pipeline loaded successfully!")
        print("=" * 70)

        self._risk_clip = None
        self._risk_clip_proc = None
        self._clip_guide = None

    def _get_risk_clip(self):
        if self._risk_clip is None:
            from transformers import CLIPModel, CLIPProcessor
            self._risk_clip_device = self.vae_device
            self._risk_clip = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14").to(self._risk_clip_device)
            self._risk_clip_proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14")
            self._risk_clip.requires_grad_(False)
            print(f"  [Risk] CLIP loaded on {self._risk_clip_device}")
        return self._risk_clip, self._risk_clip_proc, self._risk_clip_device

    def _get_clip_guide(self, unsafe_texts, threshold, lr):
        """Lazy-load CLIPSafetyGuidance. Reuses risk CLIP model (placed on vae_device)."""
        clip_model, clip_proc, clip_dev = self._get_risk_clip()
        if self._clip_guide is None:
            from safe_denoiser import CLIPSafetyGuidance
            self._clip_guide = CLIPSafetyGuidance(
                clip_model, clip_proc, unsafe_texts,
                threshold=threshold, lr=lr, device=clip_dev)
            print(f"  [CLIP-Safety] guidance loaded (τ={threshold}, lr={lr})")
        else:
            self._clip_guide.threshold = threshold
            self._clip_guide.lr = lr
        return self._clip_guide

    def _print_gpu_memory(self):
        """Print GPU memory usage for all devices."""
        print("\n[GPU Memory Usage]")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def offload_to_cpu(self):
        """Move all models to CPU to free GPU memory (e.g. before FID computation)."""
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        self.transformer.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        print("Pipeline offloaded to CPU.")

    # Version string → apply method mapping (original SAFREE only)
    VERSION_MAP = {
        "safree": "apply",
    }

    def generate(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=3.5,
        generator=None,
        # SAFREE version: None (base), "safree", "v2", "v2.1", "proj_v1", etc.
        safree_version=None,
        ss_version=None,        # None or "orig" — Semantic Surgery on prompt embeddings (independent of SAFREE)
        unsafe_concepts=None,
        safree_alpha=0.01,
        modify_clip=False,
        # Shared hyperparameters (passed via **kwargs to the apply method)
        strength=1.0,
        energy=0.9,
        svd_rank=None,
        min_strength=0.1,
        n_shuffles=10,
        neutral="a photo",
        n_pairs=20,
        radius_pct=0.95,
        gamma=None,
        n_fisher=None,
        concept_subspace=None,
        iterative_projection=True,
        true_cfg_scale=1.0,
        # Safe Denoiser / SGF parameters
        safe_denoiser_method=None,
        unsafe_images=None,
        sd_scale=0.015,
        sd_warmup_start=0.8,
        sd_warmup_end=0.4,
        sd_sigma=1.0,
        sgf_stochastic=True,           # SGFMeanFlow: paper-faithful fresh-noise re-noise
        sgf_sign="paper",              # SGFMeanFlow: 'paper' (+=) or 'intent' (-=)
        # STG (comparison method; Na et al. NeurIPS 2025) parameters
        stg_nudenet_path=None,
        stg_lr=0.1,
        stg_update_intervals=None,     # e.g. "1-3" → updates steps 1,2
        stg_update_itrs=None,          # e.g. "1-2-3" → exact indices
        stg_update_freq=1,
        stg_weight_prior=0.0,
        stg_unsafe_conf=0.01,
        stg_init_org=False,
        risk_threshold=0.3,
        renorm=False,
        # CLIPSafetyGuidance: gradient-based correction of prompt_embeds via CLIP
        clip_safety=False,
        clip_safety_threshold=0.25,
        clip_safety_lr=3.0,
        clip_safety_unsafe_texts=("nudity", "naked", "nude", "nsfw", "sexual", "erotic"),
        # Iterative Velocity Safety Guidance
        vsg_iter=False,
        vsg_iter_steps=None,
        vsg_iter_max=5,
        vsg_iter_lambda=1.0,
        vsg_iter_threshold=0.2,
        vsg_iter_scorer="clip",
        vsg_iter_unsafe_texts=("nudity", "naked", "nude", "nsfw", "sexual", "erotic"),
        vsg_iter_nudenet_ckpt=None,
        vsg_iter_verbose=False,
        vsg_iter_normalize_grad=True,
        # VESFlow score-based guidance (MeanFlow classifier guidance on mean velocity)
        score_guide=False,
        score_guide_scale=1.0,
        score_guide_kind="sigmoid",
        score_guide_scorer="nudenet",   # "nudenet" | "clip"
        score_guide_nudenet_ckpt=None,  # required when scorer="nudenet"
        score_guide_nudenet_reduce="max",          # NudeNet detection reduce: 'max' | 'mean' | 'sum'
        score_guide_nudenet_mean_threshold=0.0,    # if reduce='mean', drop confs ≤ τ before averaging
        score_guide_factor_eps=0.0,                # factor = t/(1-t+ε); enables max_t=1.0
        score_guide_grad_pool_size=1,              # z-grad pool→unpool patch size (1 = off)
        score_guide_grad_pool_mode="bilinear",     # 'bilinear' | 'nearest'
        score_guide_factor_cap=0.0,                # cap on t/(1-t) for VESFlow / VESFlow_str (0 = use factor_eps instead)
        score_guide_unsafe_texts=("nudity", "naked", "nude", "nsfw", "sexual", "erotic"),  # for scorer="clip"
        score_guide_clip_normalize="01",            # CLIP scorer: '01'|'sigmoid'|'zeroshot'|'raw'
        score_guide_clip_safe_texts=None,           # zeroshot mode: list/tuple of safe baseline texts
        score_guide_laion_head_path=None,           # override default LAION-NSFW head (for violence head, etc.)
        stg_use_clip_violence=False,                # STG with CLIP-violence scorer instead of NudeNet
        stg_clip_violence_texts=("violence", "bloody", "gore"),
        stg_clip_violence_threshold=0.0,
        score_guide_min_t=0.05,
        score_guide_max_t=0.99,
        score_guide_normalize_grad=False,
        score_guide_grad_target="x",
        score_guide_verbose=False,
        score_guide_vesflow=False,           # use VESFlow (basic: g·∇h = ∇g/(1-g))
        score_guide_vesflow_str=False,           # use VESFlowStr (stronger: ∇h = ∇g/[g(1-g)])
        score_guide_alpha=1.0,          # α for v3: divisor = g(α + (1-2α)g); α=1 ↔ v2
        score_guide_divisor_a=1.0,      # legacy (unused after v3 added)
        score_guide_divisor_eps=1e-6,
        score_guide_divisor_max=0.01,
        score_guide_skip_first_step=False,  # if True, skip score_guide on step_idx=0
        score_guide_smooth_sigma=0.0,       # randomized smoothing: image-noise std (0 = off)
        score_guide_smooth_n=4,             # randomized smoothing: MC sample count
        score_guide_grad_blur_sigma=0.0,    # gaussian blur on latent gradient (0 = off)
        score_guide_grad_blur_kernel=5,
        score_guide_ensemble_weight=0.5,    # weight for NudeNet when scorer='ensemble'
        score_guide_ensemble_unsafe_texts=("nudity", "naked", "nude", "nsfw", "sexual", "erotic"),
        # Velocity safety modifiers (CG: classifier-guided, VP: velocity projection)
        vel_safety_method=None,     # None | "cg" | "vp"
        cg_scale=0.1,
        vp_lambda=1.0,
        vp_n_iters=1,
        vel_warmup_start=1.0,
        vel_warmup_end=0.0,
        vel_unsafe_texts=("nudity", "naked", "nude", "nsfw", "sexual", "erotic"),
    ):
        """Generate image with optional SAFREE concept erasure and/or Safe Denoiser.

        Args:
            safree_version: None for base, or one of:
                "safree", "strong", "uzn", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v3",
                "proj_v1", "proj_v1.1", "proj_v1.2", "proj_v1.3",
                "proj_v2", "proj_v2.1", "proj_v3"
            safe_denoiser_method: None, "safe_denoiser", or "sgf"
            unsafe_images: path to unsafe reference images dir or .pt latents
        """

        print(f"\nGenerating image...")
        print(f"  Prompt: {prompt}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        if true_cfg_scale > 1.0:
            print(f"  True CFG scale: {true_cfg_scale}")
        print(f"  SAFREE version: {safree_version}")
        if safe_denoiser_method:
            print(f"  Safe Denoiser: {safe_denoiser_method} (scale={sd_scale})")

        # Multi-GPU: Always encode on text_encoder_device, then move to transformer_device
        # Step 1: Encode prompt on text_encoder_device
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.text_encoder_device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        # Save raw (unsafe) embeds for VP's v_unsafe transformer forward
        _orig_prompt_embeds = prompt_embeds.detach().clone()
        _orig_pooled_embeds = pooled_prompt_embeds.detach().clone()
        _orig_text_ids = text_ids.detach().clone()

        # Step 1b: Risk-aware gating via CLIP
        _skip_safety = False
        if risk_threshold > 0 and unsafe_concepts:
            from safe_denoiser import compute_risk_score_clip
            _clip, _clip_proc, _clip_dev = self._get_risk_clip()
            risk_score, _ = compute_risk_score_clip(
                prompt, unsafe_concepts, _clip, _clip_proc, device=_clip_dev)
            print(f"  [Risk] score={risk_score:.4f}, threshold={risk_threshold:.2f}")
            if risk_score < risk_threshold:
                _skip_safety = True
                print(f"  [Risk] Safe prompt — skipping safety")

        # Step 2-SS: Apply Semantic Surgery if enabled (text-side only).
        # SS is independent of SGF/STG callbacks (acts on prompt_embeds before denoising).
        ss_version = locals().get("ss_version", None)
        if ss_version not in (None, "none") and unsafe_concepts and not _skip_safety:
            from embedding_modifier import SemanticSurgeryForFLUX
            _ss = SemanticSurgeryForFLUX(
                text_encoder_2=self.text_encoder_2,
                tokenizer_2=self.tokenizer_2,
                device=self.text_encoder_device,
            )
            _apply = _ss.apply
            _result = _apply(
                prompt_embeds=prompt_embeds,
                prompt=prompt,
                concepts_to_erase=list(unsafe_concepts),
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
            if isinstance(_result, tuple):
                prompt_embeds, pooled_prompt_embeds = _result
            else:
                prompt_embeds = _result
            print(f"  [SS] {ss_version} applied to prompt_embeds")

        # Step 2: Apply SAFREE if enabled (on text_encoder_device)
        if safree_version is not None and unsafe_concepts and not _skip_safety:
            safree_obj = SAFREEForFLUX(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                text_encoder_2=self.text_encoder_2,
                tokenizer_2=self.tokenizer_2,
                device=self.text_encoder_device,
                alpha=safree_alpha,
            )
            safree_obj._precomputed = concept_subspace

            # Resolve method name
            method_name = self.VERSION_MAP.get(safree_version)
            if method_name is None:
                raise ValueError(f"Unknown safree_version: {safree_version}. "
                                 f"Available: {list(self.VERSION_MAP.keys())}")
            apply_fn = getattr(safree_obj, method_name)

            # Build kwargs based on what each method accepts
            import inspect
            sig = inspect.signature(apply_fn)
            params = sig.parameters

            kwargs = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "prompt": prompt,
                "unsafe_concepts": unsafe_concepts,
                "modify_clip": modify_clip,
            }
            # Optional params — only pass if the method accepts them
            optional = {
                "strength": strength,
                "energy_threshold": energy,
                "svd_rank": svd_rank,
                "min_strength": min_strength,
                "n_shuffles": n_shuffles,
                "neutral_concept": neutral,
                "n_pairs": n_pairs,
                "radius_percentile": radius_pct,
                "gamma": gamma,
                "n_directions": n_fisher,
            }
            for k, v in optional.items():
                if k in params:
                    kwargs[k] = v

            prompt_embeds, pooled_prompt_embeds = apply_fn(**kwargs)

        # Step 2a-renorm: Re-apply T5 LayerNorm to restore manifold statistics
        if renorm and safree_version is not None:
            ln = self.text_encoder_2.encoder.final_layer_norm
            prompt_embeds = ln(prompt_embeds.to(ln.weight.device, dtype=ln.weight.dtype))
            prompt_embeds = prompt_embeds.to(self.text_encoder_device)
            print(f"  [Renorm] T5 LayerNorm re-applied")

        # Step 2b: Build unified callback (SAFREE iterative + Safe Denoiser)
        _iter_callback = None
        _iter_inputs = None

        safree_iter_data = None
        if iterative_projection and 'safree_obj' in locals() and getattr(safree_obj, '_iter_data', None) is not None:
            safree_iter_data = safree_obj._iter_data
            print(f"  [Iterative] SAFREE projection at every denoising step")

        _stg_obj = None  # kept alive for priming after encode_prompt (STG)
        if safe_denoiser_method == "sgf" and not _skip_safety and unsafe_images:
            # Paper-faithful SGFMeanFlow port (stochastic re-noise + step-with-old-v).
            from sgf_stg_meanflow import SGFMeanFlow
            sgf_obj = SGFMeanFlow(
                vae=self.vae,
                vae_device=self.vae_device,
                transformer_device=self.transformer_device,
            )
            sgf_obj.load_unsafe_images(unsafe_images)
            _iter_callback, _iter_inputs = sgf_obj.get_callback(
                scale=sd_scale,
                warmup_start=sd_warmup_start,
                warmup_end=sd_warmup_end,
                safree_iter_data=safree_iter_data,
                stochastic_renoise=sgf_stochastic,
                sign=sgf_sign,
            )
        elif safe_denoiser_method == "stg" and not _skip_safety:
            from sgf_stg_meanflow import STGMeanFlow

            def _parse_intervals(s):
                if not s:
                    return None
                parts = [int(x) for x in str(s).split("-") if x]
                if len(parts) != 2:
                    raise ValueError(f"--stg-update-intervals expects 'lo-hi', got {s!r}")
                return tuple(parts)

            def _parse_itrs(s):
                if not s:
                    return None
                return [int(x) for x in str(s).split("-") if x]

            update_intervals = _parse_intervals(stg_update_intervals)
            update_itrs = _parse_itrs(stg_update_itrs)
            _stg_scorer = None
            if stg_use_clip_violence:
                # Mirror stg_flux.STGForFlux setup for violence: CLIP cos-sim
                # with T_unsafe = ("violence","bloody","gore"), normalize="01",
                # to make STG minimize rewards (= push away from violence).
                from transformers import CLIPModel, CLIPProcessor
                from safe_denoiser import CLIPVelocityScorer
                _clip_dev = str(self.vae_device)
                _clip_m = CLIPModel.from_pretrained(
                    "openai/clip-vit-large-patch14").to(_clip_dev).eval()
                for _p in _clip_m.parameters():
                    _p.requires_grad_(False)
                _clip_p = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-large-patch14")
                _stg_scorer = CLIPVelocityScorer(
                    _clip_m, _clip_p,
                    list(stg_clip_violence_texts),
                    device=_clip_dev,
                    normalize="01",
                )
                print(f"  [STG-violence] CLIP scorer on {_clip_dev} "
                      f"texts={list(stg_clip_violence_texts)} normalize=01")
            _stg_obj = STGMeanFlow(
                transformer=self.transformer,
                vae=self.vae,
                transformer_device=self.transformer_device,
                vae_device=self.vae_device,
                nudenet_path=None if _stg_scorer is not None else stg_nudenet_path,
                nudenet_device=self.transformer_device,
                conf=stg_unsafe_conf,
                scorer=_stg_scorer,
            )
            _iter_callback, _iter_inputs = _stg_obj.get_callback(
                lr=stg_lr,
                update_itrs=update_itrs,
                update_intervals=update_intervals,
                update_freq=stg_update_freq,
                weight_prior=stg_weight_prior,
                init_org=stg_init_org,
            )
        elif safe_denoiser_method and unsafe_images:
            from safe_denoiser import SafeDenoiserForFLUX
            sd_obj = SafeDenoiserForFLUX(vae=self.vae, device=self.vae_device)
            sd_obj.load_unsafe_images(unsafe_images)
            _iter_callback, _iter_inputs = sd_obj.get_callback(
                method=safe_denoiser_method, scale=sd_scale,
                warmup_start=sd_warmup_start, warmup_end=sd_warmup_end,
                sigma=sd_sigma, safree_iter_data=safree_iter_data,
            )
        elif safree_iter_data is not None:
            vd = safree_iter_data
            def _step_callback(pipeline, step_idx, timestep, callback_kwargs):
                pe = callback_kwargs["prompt_embeds"]
                text_e = pe.squeeze(0).float()
                _V = vd["V_r"].to(text_e.device, dtype=text_e.dtype)
                _I = vd["I_m_P_r"].to(text_e.device, dtype=text_e.dtype)
                _m = vd["attn_mask"].to(text_e.device)
                t_low = text_e @ _V
                t_safe = (_I @ t_low.T).T
                delta = vd["strength"] * (t_safe - t_low)
                t_proj = text_e + (delta @ _V.T)
                mask = _m.unsqueeze(1)
                merged = text_e * (1.0 - mask) + t_proj * mask
                callback_kwargs["prompt_embeds"] = merged.unsqueeze(0).to(pe.dtype)
                return callback_kwargs
            _iter_callback = _step_callback
            _iter_inputs = ["prompt_embeds"]

        # Step 2b-iter-vsg: Iterative Velocity Safety Guidance
        if vsg_iter and not _skip_safety:
            from safe_denoiser import (IterativeVSG, CLIPVelocityScorer,
                                           NudeNetYOLOVelocityScorer)

            if vsg_iter_scorer == "clip":
                _clip, _clip_proc, _clip_dev = self._get_risk_clip()
                _scorer = CLIPVelocityScorer(
                    _clip, _clip_proc, list(vsg_iter_unsafe_texts), device=_clip_dev)
            elif vsg_iter_scorer == "nudenet_yolo":
                if vsg_iter_nudenet_ckpt is None:
                    raise ValueError("vsg_iter_scorer='nudenet_yolo' requires vsg_iter_nudenet_ckpt")
                _scorer = NudeNetYOLOVelocityScorer(
                    vsg_iter_nudenet_ckpt, device=self.vae_device)
            else:
                raise ValueError(f"Unknown vsg_iter_scorer: {vsg_iter_scorer}")

            _iter_vsg = IterativeVSG(
                vae=self.vae, scorer=_scorer,
                threshold=vsg_iter_threshold, max_iters=vsg_iter_max,
                lambda_base=vsg_iter_lambda, verbose=vsg_iter_verbose,
                normalize_grad=vsg_iter_normalize_grad)

            _apply_steps = (set(vsg_iter_steps) if vsg_iter_steps is not None else None)
            _prev_callback_iv = _iter_callback
            _prev_inputs_iv = _iter_inputs

            def _iter_vsg_callback(pipeline, step_idx, timestep, callback_kwargs):
                if _prev_callback_iv is not None:
                    callback_kwargs = _prev_callback_iv(
                        pipeline, step_idx, timestep, callback_kwargs)

                if _apply_steps is not None and step_idx not in _apply_steps:
                    return callback_kwargs

                latents = callback_kwargs["latents"]
                noise_pred = callback_kwargs["noise_pred"]

                scheduler = pipeline.scheduler
                sigmas = getattr(scheduler, "sigmas", None)
                if sigmas is not None:
                    sigma_cur = float(sigmas[step_idx])
                    sigma_next = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
                else:
                    sigma_cur = float(timestep) / 1000.0
                    sigma_next = 0.0

                dt = sigma_next - sigma_cur
                x_t = latents - dt * noise_pred

                v_new, hist = _iter_vsg.correct(noise_pred, x_t, sigma_cur)

                latents_new = x_t + dt * v_new
                callback_kwargs["latents"] = latents_new.to(latents.dtype)
                callback_kwargs["noise_pred"] = v_new.to(noise_pred.dtype)

                print(f"  [IterVSG] step={step_idx}, σ={sigma_cur:.4f}, "
                      f"iters={len(hist)}, scores={[f'{s:.3f}' for s in hist]}")
                return callback_kwargs

            _iter_callback = _iter_vsg_callback
            _iter_inputs = list(set((_prev_inputs_iv or []) + ["latents", "noise_pred"]))
            _apply_desc = "all" if _apply_steps is None else sorted(_apply_steps)
            print(f"  [IterVSG] enabled, steps={_apply_desc}, "
                  f"max_iter={vsg_iter_max}, λ={vsg_iter_lambda}, "
                  f"τ={vsg_iter_threshold}, scorer={vsg_iter_scorer}")

        # Step 2b-score-guide: VESFlow score-based guidance
        # u'(x_t, 0, t) = u − λ · ∇_{x_t} log[p(unsafe|z_t)/p(safe|z_t)]
        if score_guide and not _skip_safety:
            from safe_denoiser import (
                NudeNetYOLOVelocityScorer, CLIPVelocityScorer)
            if score_guide_scorer == "nudenet":
                if score_guide_nudenet_ckpt is None:
                    raise ValueError("score_guide scorer='nudenet' requires score_guide_nudenet_ckpt")
                _sg_scorer = NudeNetYOLOVelocityScorer(
                    score_guide_nudenet_ckpt, device=str(self.vae_device),
                    reduce=score_guide_nudenet_reduce,
                    mean_threshold=score_guide_nudenet_mean_threshold)
            elif score_guide_scorer == "clip":
                _clip_m, _clip_p, _clip_d = self._get_risk_clip()
                _sg_scorer = CLIPVelocityScorer(
                    _clip_m, _clip_p, list(score_guide_unsafe_texts),
                    device=str(_clip_d),
                    normalize=score_guide_clip_normalize,
                    safe_texts=(list(score_guide_clip_safe_texts)
                                if score_guide_clip_safe_texts else None),
                )
            elif score_guide_scorer == "q16":
                from safe_denoiser import Q16VelocityScorer
                _sg_scorer = Q16VelocityScorer(device=str(self.vae_device))
                print(f"  [Q16] CLIP ViT-L/14 + soft prompts on {self.vae_device}")
            elif score_guide_scorer == "laion_nsfw":
                from safe_denoiser import LAIONNudityVelocityScorer
                # Cache scorer across calls — the CLIP-L instantiation costs ~12s.
                _cache_key = ("laion_nsfw", str(self.vae_device), score_guide_laion_head_path)
                if getattr(self, "_score_guide_scorer_cache", None) is not None \
                        and self._score_guide_scorer_cache_key == _cache_key:
                    _sg_scorer = self._score_guide_scorer_cache
                else:
                    _sg_scorer = LAIONNudityVelocityScorer(
                        device=str(self.vae_device),
                        head_path=score_guide_laion_head_path)
                    self._score_guide_scorer_cache = _sg_scorer
                    self._score_guide_scorer_cache_key = _cache_key
                    _tag = score_guide_laion_head_path or "default LAION-NSFW head"
                    print(f"  [LAION-NSFW] CLIP ViT-L/14 + head={_tag} on {self.vae_device} (cached)")
            elif score_guide_scorer == "ensemble":
                if score_guide_nudenet_ckpt is None:
                    raise ValueError("ensemble requires score_guide_nudenet_ckpt")
                _nud = NudeNetYOLOVelocityScorer(
                    score_guide_nudenet_ckpt, device=str(self.vae_device))
                _clip_m, _clip_p, _clip_d = self._get_risk_clip()
                _clp = CLIPVelocityScorer(
                    _clip_m, _clip_p, list(score_guide_ensemble_unsafe_texts),
                    device=str(_clip_d),
                )
                from safe_denoiser import EnsembleScorer
                w = score_guide_ensemble_weight
                _sg_scorer = EnsembleScorer([_nud, _clp], weights=[w, 1.0 - w])
                print(f"  [Ensemble] NudeNet w={w:.2f}, CLIP w={1-w:.2f}")
            else:
                raise ValueError(f"Unknown score_guide_scorer: {score_guide_scorer}")
            # Optional randomized smoothing wrapper
            if score_guide_smooth_sigma > 0 and score_guide_smooth_n > 0:
                from safe_denoiser import SmoothedScorer
                _sg_scorer = SmoothedScorer(_sg_scorer,
                                            noise_sigma=score_guide_smooth_sigma,
                                            n_samples=score_guide_smooth_n)
                print(f"  [SmoothedScorer] σ={score_guide_smooth_sigma} n={score_guide_smooth_n}")
            if score_guide_vesflow_str:
                from safe_denoiser import VESFlowStr
                _score_guide = VESFlowStr(
                    vae=self.vae, scorer=_sg_scorer,
                    scale=score_guide_scale,
                    kind=score_guide_kind, sigma_data=None,
                    min_t=score_guide_min_t, max_t=score_guide_max_t,
                    normalize_grad=score_guide_normalize_grad,
                    latent_packed=True, verbose=score_guide_verbose,
                    divisor_max=score_guide_divisor_max,
                    grad_blur_sigma=score_guide_grad_blur_sigma,
                    grad_blur_kernel=score_guide_grad_blur_kernel,
                    factor_eps=score_guide_factor_eps,
                    grad_pool_size=score_guide_grad_pool_size,
                    grad_pool_mode=score_guide_grad_pool_mode,
                    factor_cap=score_guide_factor_cap,
                )
            elif score_guide_vesflow:
                from safe_denoiser import VESFlow
                _score_guide = VESFlow(
                    vae=self.vae, scorer=_sg_scorer,
                    scale=score_guide_scale,
                    kind=score_guide_kind, sigma_data=None,
                    min_t=score_guide_min_t, max_t=score_guide_max_t,
                    normalize_grad=score_guide_normalize_grad,
                    latent_packed=True, verbose=score_guide_verbose,
                    divisor_max=score_guide_divisor_max,
                    grad_blur_sigma=score_guide_grad_blur_sigma,
                    grad_blur_kernel=score_guide_grad_blur_kernel,
                    factor_eps=score_guide_factor_eps,
                    grad_pool_size=score_guide_grad_pool_size,
                    grad_pool_mode=score_guide_grad_pool_mode,
                    factor_cap=score_guide_factor_cap,
                )
            else:
                raise ValueError(
                    "score_guide requires score_guide_vesflow=True (VESFlow basic) "
                    "or score_guide_vesflow_str=True (VESFlowStr stronger).")

            _prev_cb_sg = _iter_callback
            _prev_in_sg = _iter_inputs

            def _score_guide_callback(pipeline, step_idx, timestep, callback_kwargs):
                if _prev_cb_sg is not None:
                    callback_kwargs = _prev_cb_sg(pipeline, step_idx, timestep, callback_kwargs)

                if score_guide_skip_first_step and step_idx == 0:
                    return callback_kwargs

                latents_post = callback_kwargs["latents"]
                u = callback_kwargs["noise_pred"]

                sched = pipeline.scheduler
                sigmas = getattr(sched, "sigmas", None)
                if sigmas is not None and step_idx < len(sigmas):
                    sigma_cur = float(sigmas[step_idx])
                    sigma_next = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
                else:
                    sigma_cur = float(timestep) / 1000.0
                    sigma_next = 0.0
                dt = sigma_next - sigma_cur

                # reconstruct pre-step x_t
                x_t_pre = latents_post - dt * u

                # VESFlow / VESFlowStr: u' = u − λ · ∇_{x_t} log_ratio
                u_new = _score_guide.apply(u, x_t_pre, sigma_cur)

                # re-step with u_new
                latents_new = x_t_pre + dt * u_new
                callback_kwargs["latents"] = latents_new.to(latents_post.dtype)
                callback_kwargs["noise_pred"] = u_new.to(u.dtype)
                return callback_kwargs

            _iter_callback = _score_guide_callback
            _iter_inputs = list(set((_prev_in_sg or []) + ["latents", "noise_pred"]))
            print(f"  [ScoreGuide] enabled, scale={score_guide_scale}, "
                  f"kind={score_guide_kind}, normalize_grad={score_guide_normalize_grad}")

        # Step 2b-vel-safety: CG or VP velocity modifier (MF: flow-matching x_0=x_t-σv)
        if vel_safety_method is not None and not _skip_safety:
            import math
            from safe_denoiser import CGVelocityModifier, VPVelocityModifier

            _cg_mod = None
            _vp_mod = None
            if vel_safety_method == "cg":
                _clip, _clip_proc, _clip_dev = self._get_risk_clip()
                _cg_mod = CGVelocityModifier(
                    vae=self.vae, clip_model=_clip, clip_processor=_clip_proc,
                    unsafe_concepts=list(vel_unsafe_texts),
                    scale=cg_scale,
                    vae_device=str(self.vae_device),
                    clip_device=str(_clip_dev),
                )
            elif vel_safety_method == "vp":
                _orig_pe_tr = _orig_prompt_embeds.to(self.transformer_device, dtype=self.dtype)
                _orig_ppe_tr = _orig_pooled_embeds.to(self.transformer_device, dtype=self.dtype)
                _orig_tid_tr = _orig_text_ids.to(self.transformer_device, dtype=self.dtype)
                _vp_mod = VPVelocityModifier(
                    transformer=self.transformer,
                    orig_prompt_embeds=_orig_pe_tr,
                    orig_pooled_embeds=_orig_ppe_tr,
                    text_ids=_orig_tid_tr,
                    guidance_scale=guidance_scale,
                    n_iters=vp_n_iters,
                    lambda_schedule=[vp_lambda] * vp_n_iters,
                    transformer_device=str(self.transformer_device),
                )
            else:
                raise ValueError(f"Unknown vel_safety_method: {vel_safety_method}")

            _prev_callback_vs = _iter_callback
            _prev_inputs_vs = _iter_inputs
            _vae_scale = self.vae.config.scaling_factor
            _vae_shift = self.vae.config.shift_factor

            # closure state: pre-step latents (MF uses flow-matching; initial = raw latents)
            _vel_state = {"x_t_pre": None, "initial_latents": None}

            def _vel_safety_callback(pipeline, step_idx, timestep, callback_kwargs):
                if _prev_callback_vs is not None:
                    callback_kwargs = _prev_callback_vs(
                        pipeline, step_idx, timestep, callback_kwargs)

                latents_post = callback_kwargs["latents"]
                noise_pred = callback_kwargs["noise_pred"]

                sched = pipeline.scheduler
                sigmas = getattr(sched, "sigmas", None)
                if sigmas is not None and step_idx < len(sigmas):
                    sigma_cur = float(sigmas[step_idx])
                    sigma_next = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
                else:
                    sigma_cur = float(timestep) / 1000.0
                    sigma_next = 0.0
                dt = sigma_next - sigma_cur

                # MF flow-matching Euler: x_t_post = x_t_pre + dt · v  →  x_t_pre = x_t_post - dt·v
                if step_idx == 0 and _vel_state["initial_latents"] is not None:
                    x_t_pre = _vel_state["initial_latents"].to(noise_pred.device,
                                                               dtype=noise_pred.dtype)
                else:
                    x_t_pre = latents_post - dt * noise_pred

                if _cg_mod is not None:
                    v_new = _cg_mod.apply(
                        v_packed=noise_pred, x_t_packed=x_t_pre,
                        sigma_t=sigma_cur, sigma_next=sigma_next,
                        scaling_factor=_vae_scale, shift_factor=_vae_shift,
                    )
                else:
                    # MF transformer needs timestep/1000 + timestep2=(t-r)/1000.
                    sched_t = getattr(sched, "timesteps", None)
                    if sched_t is not None and step_idx < len(sched_t):
                        t_val = float(sched_t[step_idx])
                        r_val = float(sched_t[step_idx + 1]) if step_idx + 1 < len(sched_t) else 0.0
                    else:
                        t_val = float(timestep) if not torch.is_tensor(timestep) else float(timestep.item())
                        r_val = 0.0
                    dev = noise_pred.device
                    dt_type = noise_pred.dtype
                    ts_div = torch.tensor([t_val / 1000.0], device=dev, dtype=dt_type)
                    ts2 = torch.tensor([(t_val - r_val) / 1000.0], device=dev, dtype=dt_type)
                    v_new = _vp_mod.apply(noise_pred, x_t_pre, ts_div,
                                          extra_tf_kwargs={"timestep2": ts2})

                # Re-step Euler: latents_post_new = x_t_pre + dt · v_new
                latents_post_new = x_t_pre + dt * v_new
                callback_kwargs["latents"] = latents_post_new.to(latents_post.dtype)
                callback_kwargs["noise_pred"] = v_new.to(noise_pred.dtype)

                _vel_state["x_t_pre"] = latents_post_new.detach()
                return callback_kwargs

            _iter_callback = _vel_safety_callback
            _iter_inputs = list(set((_prev_inputs_vs or []) + ["latents", "noise_pred"]))
            print(f"  [VelSafety] method={vel_safety_method}, "
                  f"cg_scale={cg_scale}, vp_λ={vp_lambda}, vp_iters={vp_n_iters}")

        # Step 2c: Build negative prompt embeds for true CFG
        # Two paths:
        #   (a) SAFREE active  → use SAFREE's negative-concept subspace
        #   (b) SAFREE off     → directly use `unsafe_concepts` as the negative prompt
        # Both yield  v = scale·v(prompt) − (scale−1)·v(neg)  inside the pipeline.
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        do_true_cfg = true_cfg_scale > 1.0 and bool(unsafe_concepts)
        if do_true_cfg:
            if safree_version is not None:
                negative_prompt_space = SAFREEForFLUX(
                    text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                    text_encoder_2=self.text_encoder_2, tokenizer_2=self.tokenizer_2,
                    device=self.text_encoder_device,
                ).get_negative_prompt_space(unsafe_concepts)
                neg_prompt_str = ", ".join(negative_prompt_space)
            else:
                # Direct negative-concept CFG without SAFREE projection
                neg_prompt_str = ", ".join(unsafe_concepts)
            negative_prompt_embeds, negative_pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=neg_prompt_str, prompt_2=neg_prompt_str,
                device=self.text_encoder_device,
                num_images_per_prompt=1, max_sequence_length=256,
            )
            print(f"  [True CFG] scale={true_cfg_scale}  negative_prompt: {neg_prompt_str[:80]}...")

        # Step 3: Move ALL tensors to transformer device for generation
        transformer_dev = torch.device(self.transformer_device)
        prompt_embeds = prompt_embeds.to(transformer_dev, dtype=self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dev, dtype=self.dtype)
        text_ids = text_ids.to(transformer_dev, dtype=self.dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dev, dtype=self.dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dev, dtype=self.dtype)

        # Step 4: Pre-create latents on transformer device
        # This is critical for multi-GPU: latents must be on same device as transformer
        batch_size = 1
        num_channels_latents = self.transformer.config.in_channels // 4
        vae_scale_factor = self.pipeline.vae_scale_factor

        latent_height = 2 * (int(height) // vae_scale_factor)
        latent_width = 2 * (int(width) // vae_scale_factor)

        latent_shape = (batch_size, num_channels_latents, latent_height, latent_width)
        latents = torch.randn(latent_shape, generator=generator, device=transformer_dev, dtype=self.dtype)

        # Pack latents (FLUX-style packing)
        latents = latents.view(batch_size, num_channels_latents, latent_height // 2, 2, latent_width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (latent_height // 2) * (latent_width // 2), num_channels_latents * 4)

        # Step 4a-bis: Prime STG with static conditioning now that all tensors
        # are on the transformer device (STG re-runs the transformer w/ grad).
        if _stg_obj is not None:
            from sgf_stg_meanflow import prepare_latent_image_ids
            _stg_latent_ids = prepare_latent_image_ids(
                self.pipeline, batch_size, height, width, transformer_dev, self.dtype)
            _stg_guidance = torch.full(
                [1], guidance_scale, device=transformer_dev, dtype=torch.float32
            ).expand(batch_size).to(self.dtype)
            _stg_obj.prime(
                latent_image_ids=_stg_latent_ids,
                text_ids=text_ids,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance=_stg_guidance,
                prompt_embeds_given=prompt_embeds,
            )

        # Step 4b: CLIPSafetyGuidance one-shot correction of prompt_embeds
        if clip_safety and not _skip_safety:
            clip_guide = self._get_clip_guide(
                list(clip_safety_unsafe_texts), clip_safety_threshold, clip_safety_lr)
            latent_image_ids = self.pipeline._prepare_latent_image_ids(
                batch_size, latent_height, latent_width, transformer_dev, self.dtype)
            # MeanFlow: predicts mean velocity on [r, t]. One-shot: t=1.0, r=0.
            sigma_max = 1.0
            x_t_init = latents
            timestep_t = torch.tensor([1.0 / 1000.0], device=transformer_dev, dtype=self.dtype)
            timestep2_t = torch.tensor([1.0 / 1000.0], device=transformer_dev, dtype=self.dtype)
            guidance_t = torch.tensor([guidance_scale], device=transformer_dev, dtype=self.dtype)
            try:
                pe_corr, cs_score = clip_guide.correct_embedding(
                    prompt_embeds, x_t_init, sigma_max,
                    self.transformer, self.vae,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance_t,
                    timestep=timestep_t,
                    transformer_extra_kwargs={"timestep2": timestep2_t},
                )
                prompt_embeds = pe_corr.to(transformer_dev, dtype=self.dtype)
                print(f"  [CLIP-Safety] score={cs_score.item():.4f}, "
                      f"{'APPLIED' if cs_score.item() > clip_safety_threshold else 'skipped'}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [CLIP-Safety] OOM — skipping correction")

        # Expose initial latents to vel_safety callback for step 0's x_t_pre.
        # MF does not pre-scale by sigma_data (unlike sCM), so pass as-is.
        if vel_safety_method is not None and not _skip_safety:
            _vel_state["initial_latents"] = latents.detach()

        # Step 5: Generate with embeddings and pre-created latents
        # Get latents output (skip VAE decoding in pipeline)
        result = self.pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            true_cfg_scale=true_cfg_scale if do_true_cfg else 1.0,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,  # Pass pre-created latents on correct device
            max_sequence_length=256,
            output_type="latent",  # Return latents, handle VAE separately for multi-GPU
            callback_on_step_end=_iter_callback,
            callback_on_step_end_tensor_inputs=_iter_inputs,
        )

        # Step 6: Decode latents on VAE device (multi-GPU aware)
        latent_output = result.images  # This is actually latents when output_type="latent"

        # Move latents to VAE device
        vae_dev = torch.device(self.vae_device)
        latent_output = latent_output.to(vae_dev)

        # Unpack latents (FLUX style)
        latent_output = self.pipeline._unpack_latents(latent_output, height, width, self.pipeline.vae_scale_factor)

        # Decode with VAE
        latent_output = (latent_output / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            image = self.vae.decode(latent_output, return_dict=False)[0]

        # Post-process image
        image = self.pipeline.image_processor.postprocess(image, output_type="pil")

        # Print memory after generation
        self._print_gpu_memory()

        return image[0]


def create_comparison_grid(images, labels, images_per_row=3):
    """Create a grid of images with labels."""
    from PIL import ImageDraw, ImageFont

    # Get dimensions
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)

    # Add space for labels
    label_height = 40
    cell_height = max_height + label_height

    # Calculate grid dimensions
    n_images = len(images)
    n_cols = min(n_images, images_per_row)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create canvas
    grid_width = n_cols * max_width
    grid_height = n_rows * cell_height
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // n_cols
        col = idx % n_cols
        x = col * max_width
        y = row * cell_height

        # Paste image
        grid.paste(img, (x, y + label_height))

        # Draw label
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (max_width - text_width) // 2
        draw.text((text_x, y + 10), label, fill=(0, 0, 0), font=font)

    return grid


def create_safree_comparison_grid(images_no_safree, images_safree, step_labels, safree_mode="SAFREE"):
    """
    Create a 2-row comparison grid.
    Row 1: Without SAFREE
    Row 2: With SAFREE
    """
    from PIL import ImageDraw, ImageFont

    n_cols = len(step_labels)

    # Get dimensions from first image
    img_width, img_height = images_no_safree[0].size

    # Layout settings
    label_height = 40
    row_label_width = 120
    cell_height = img_height + label_height

    # Create canvas
    grid_width = row_label_width + n_cols * img_width
    grid_height = 2 * cell_height + label_height  # Extra label height for column headers
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        font_bold = font

    draw = ImageDraw.Draw(grid)

    # Draw column headers (step numbers)
    for col, label in enumerate(step_labels):
        x = row_label_width + col * img_width + img_width // 2
        text_bbox = draw.textbbox((0, 0), label, font=font_bold)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((x - text_width // 2, 10), label, fill=(0, 0, 0), font=font_bold)

    # Row labels and images
    row_labels = ["No SAFREE", safree_mode]
    all_images = [images_no_safree, images_safree]

    for row_idx, (row_label, row_images) in enumerate(zip(row_labels, all_images)):
        y_base = label_height + row_idx * cell_height

        # Draw row label
        text_bbox = draw.textbbox((0, 0), row_label, font=font_bold)
        text_height = text_bbox[3] - text_bbox[1]
        label_y = y_base + (cell_height - text_height) // 2
        draw.text((10, label_y), row_label, fill=(0, 0, 0), font=font_bold)

        # Draw images
        for col, img in enumerate(row_images):
            x = row_label_width + col * img_width
            grid.paste(img, (x, y_base))

    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="MeanFlow with SAFREE (Multi-GPU Support)")

    # Model paths
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="Freepik/flux.1-lite-8B",
        help="Pretrained FLUX model path"
    )
    parser.add_argument(
        "--meanflow-checkpoint",
        type=str,
        default=None,
        help="MeanFlow checkpoint path (if None, uses default from HuggingFace)"
    )

    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (MeanFlow: 4 recommended)")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)

    # Step comparison mode
    parser.add_argument("--compare-steps", action="store_true",
                       help="Compare multiple steps (1,2,4,6,8,16)")
    parser.add_argument("--compare-safree", action="store_true",
                       help="Compare SAFREE vs no-SAFREE across steps (2-row grid)")
    parser.add_argument("--step-list", nargs="+", type=int, default=[1, 2, 4, 6, 8, 16],
                       help="Steps to compare when --compare-steps is enabled")
    parser.add_argument("--guidance-list", nargs="+", type=float, default=None,
                       help="Guidance scales to compare (overrides --guidance-scale in comparison modes)")

    # SAFREE version selector
    parser.add_argument("--safree-version", type=str, default=None,
                       choices=[None, "safree", "strong", "uzn", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v3",
                                "proj_v1", "proj_v1.1", "proj_v1.2", "proj_v1.3",
                                "proj_v2", "proj_v2.1", "proj_v3"],
                       help="SAFREE variant to use (None = base generation)")
    parser.add_argument("--unsafe-concepts", nargs="+", default=["nudity", "naked", "nude"])
    parser.add_argument("--safree-alpha", type=float, default=0.01)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0,
                       help="True CFG scale (>1.0 enables CFG with concept as negative prompt)")
    parser.add_argument("--modify-clip", action="store_true")
    # Hyperparameters
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--energy", type=float, default=0.9)
    parser.add_argument("--svd-rank", type=int, default=None)
    parser.add_argument("--min-strength", type=float, default=0.1)
    parser.add_argument("--n-shuffles", type=int, default=10)
    parser.add_argument("--neutral", type=str, default="a photo")
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument("--radius-pct", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--n-fisher", type=int, default=None)
    parser.add_argument("--concept-subspace", type=str, default=None)
    parser.add_argument("--no-iterative", action="store_true")

    # Safe Denoiser / SGF parameters
    parser.add_argument("--safe-denoiser", type=str, default=None,
                       choices=[None, "safe_denoiser", "sgf"])
    parser.add_argument("--unsafe-images", type=str, default=None)
    parser.add_argument("--sd-scale", type=float, default=0.015)
    parser.add_argument("--sd-warmup-start", type=float, default=0.8)
    parser.add_argument("--sd-warmup-end", type=float, default=0.4)
    parser.add_argument("--sd-sigma", type=float, default=1.0)

    # Multi-GPU parameters
    parser.add_argument("--text-encoder-device", type=str, default="cuda:0",
                       help="Device for text encoders (CLIP + T5, ~11GB)")
    parser.add_argument("--transformer-device", type=str, default="cuda:1",
                       help="Device for transformer (largest, ~16GB)")
    parser.add_argument("--vae-device", type=str, default="cuda:2",
                       help="Device for VAE (~1GB)")
    parser.add_argument("--single-gpu", action="store_true",
                       help="Use single GPU (cuda:0) for all components")

    # Output
    parser.add_argument("--output-dir", type=str, default="output_vis_samplings/outputs_meanflow_safree")

    return parser.parse_args()


def main():
    args = parse_args()

    # Print GPU info
    print(f"\n[System Info]")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

    # Single GPU mode
    if args.single_gpu:
        args.text_encoder_device = "cuda:0"
        args.transformer_device = "cuda:0"
        args.vae_device = "cuda:0"
        print("\n[Single GPU mode] All components on cuda:0")

    # Find MeanFlow checkpoint
    if args.meanflow_checkpoint is None:
        # Try to find from HuggingFace cache
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--Alibaba-DAMO-Academy--T2I-Distill")
        if os.path.exists(hf_cache):
            snapshots_dir = os.path.join(hf_cache, "snapshots")
            if os.path.exists(snapshots_dir):
                for snapshot in os.listdir(snapshots_dir):
                    checkpoint_path = os.path.join(snapshots_dir, snapshot, "flux1lite_meanflow.pt")
                    if os.path.exists(checkpoint_path):
                        args.meanflow_checkpoint = checkpoint_path
                        print(f"Found MeanFlow checkpoint: {checkpoint_path}")
                        break

    if args.meanflow_checkpoint is None:
        print("Warning: No MeanFlow checkpoint found. Using base model weights.")

    # Create pipeline with Multi-GPU support
    pipe = MeanFlowSAFREEPipeline(
        pretrained_model_path=args.pretrained_model,
        meanflow_checkpoint_path=args.meanflow_checkpoint,
        text_encoder_device=args.text_encoder_device,
        transformer_device=args.transformer_device,
        vae_device=args.vae_device,
        dtype=torch.bfloat16,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Helper: build generate kwargs from args
    def gen_kwargs(version=None, generator=None, num_steps=None, guidance=None):
        return dict(
            prompt=args.prompt, height=args.height, width=args.width,
            num_inference_steps=num_steps or args.steps,
            guidance_scale=guidance or args.guidance_scale,
            generator=generator,
            safree_version=version,
            unsafe_concepts=args.unsafe_concepts, safree_alpha=args.safree_alpha,
            modify_clip=args.modify_clip,
            strength=args.strength, energy=args.energy, svd_rank=args.svd_rank,
            min_strength=args.min_strength, n_shuffles=args.n_shuffles,
            neutral=args.neutral, n_pairs=args.n_pairs,
            radius_pct=args.radius_pct, gamma=args.gamma, n_fisher=args.n_fisher,
            concept_subspace=args.concept_subspace,
            iterative_projection=not args.no_iterative,
            true_cfg_scale=args.true_cfg_scale,
            safe_denoiser_method=args.safe_denoiser,
            unsafe_images=args.unsafe_images,
            sd_scale=args.sd_scale,
            sd_warmup_start=args.sd_warmup_start,
            sd_warmup_end=args.sd_warmup_end,
            sd_sigma=args.sd_sigma,
        )

    # SAFREE comparison mode (2-row: no SAFREE vs SAFREE)
    if args.compare_safree:
        iterate = args.guidance_list or args.step_list
        images_base, images_safree, labels = [], [], []

        for val in iterate:
            if args.guidance_list:
                ns, gs = args.steps, val
                labels.append(f"G={val}")
            else:
                ns, gs = val, args.guidance_scale
                labels.append(f"Step {val}")

            gen = torch.Generator(device=args.transformer_device).manual_seed(args.seed) if args.seed else None
            images_base.append(pipe.generate(**gen_kwargs(None, gen, ns, gs)))
            gen = torch.Generator(device=args.transformer_device).manual_seed(args.seed) if args.seed else None
            images_safree.append(pipe.generate(**gen_kwargs(args.safree_version or "safree", gen, ns, gs)))

        grid = create_safree_comparison_grid(images_base, images_safree, labels, args.safree_version or "safree")
        seed_str = f"_seed{args.seed}" if args.seed else ""
        grid.save(os.path.join(args.output_dir, f"safree_comparison{seed_str}.png"))

    else:
        # Single generation mode
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=args.transformer_device).manual_seed(args.seed)

        image = pipe.generate(**gen_kwargs(args.safree_version, generator))

        suffix = f"_{args.safree_version}" if args.safree_version else ""
        seed_str = f"_seed{args.seed}" if args.seed else ""
        filename = f"meanflow_step{args.steps}{suffix}{seed_str}.png"
        output_path = os.path.join(args.output_dir, filename)

        image.save(output_path)
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
