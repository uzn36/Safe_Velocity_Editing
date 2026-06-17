"""Embedding-space safety modifiers applied to prompt_embeds BEFORE denoising.

Two orthogonal methods (each callable directly on a prompt embedding):

  - SAFREEForFLUX        (Yoon et al. 2024)
        Project prompt_embeds onto the subspace orthogonal to a set of
        unsafe-concept embeddings. Supports T5 + CLIP encoders.

  - SemanticSurgeryForFLUX (Xiong et al. 2025, "Semantic Surgery")
        Subtract a learned linear combination of unsafe-concept embeddings
        from prompt_embeds (sigmoid-gated by per-token similarity).

Both modifiers are SAFREE-style: they leave the diffusion model untouched
and act only on the conditioning embeddings, so they compose with any
denoising-step guidance (SGF, STG, VESFlow).
"""
import os
import torch
import torch.nn.functional as F



# ============================================================================
# SAFREE (Yoon et al. 2024) — projection onto orthogonal subspace
# ============================================================================

def projection_matrix(E):
    """Projection matrix onto subspace spanned by columns of E.  P = E (E^T E)^+ E^T"""
    P = E.float() @ torch.pinverse((E.T.float() @ E.float()).float()) @ E.T.float()
    return P

def safree_projection(input_embeddings, p_emb, masked_input_subspace_projection,
                      concept_subspace_projection, alpha=0.01, max_length=512,
                      token_offset=0, debug=False):
    """
    Apply SAFREE token-level orthogonal projection.

    This is the core SAFREE algorithm from official implementation:
    modified_stable_diffusion_pipeline.py:57-99

    Args:
        input_embeddings: [2, seq_len, dim] - uncond + cond embeddings
        p_emb: [n_tokens, dim] - masked prompt embeddings (from _masked_encode_prompt)
        masked_input_subspace_projection: projection matrix of masked input (ms)
        concept_subspace_projection: projection matrix of toxic concepts (cs)
        alpha: threshold for identifying toxic tokens
        max_length: max sequence length
        token_offset: offset for mask placement (0 for T5 which has no BOS,
                      1 for CLIP/SDv3 which has BOS at position 0)

    Returns:
        Modified embeddings with toxic tokens projected away from concept subspace
    """
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection
    device = ie.device
    (n_t, dim) = p_emb.shape

    # Compute I - concept_subspace_projection (orthogonal complement)
    I_m_cs = torch.eye(dim).to(device).to(cs.dtype) - cs

    # Compute distance of each token from toxic subspace
    # dist = ||(I - P_concept) @ token||
    dist_vec = I_m_cs @ p_emb.T.to(cs.dtype)
    dist_p_emb = torch.norm(dist_vec, dim=0)

    # Compute leave-one-out means for adaptive thresholding
    means = []
    for i in range(n_t):
        mean_without_i = torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i+1:])))
        means.append(mean_without_i)
    mean_dist = torch.tensor(means).to(device)

    # Identify safe (1) and unsafe/trigger (0) tokens
    rm_vector = (dist_p_emb < (1. + alpha) * mean_dist).float()
    inv_vector = (dist_p_emb >= (1. + alpha) * mean_dist).float()
    n_removed = n_t - rm_vector.sum()

    print(f"  [SAFREE] Among {n_t} tokens, removing {int(n_removed)} toxic tokens.")

    # Create mask matching sequence length
    # token_offset=0 for T5 (no BOS, content starts at position 0)
    # token_offset=1 for CLIP/SDv3 (BOS at position 0, content starts at position 1)
    ones_tensor = torch.ones(max_length).to(device)
    start = token_offset
    end = min(token_offset + n_t, max_length)
    n_fit = end - start
    ones_tensor[start:end] = rm_vector[:n_fit]
    ones_tensor = ones_tensor.unsqueeze(1)
    inverse_tensor = torch.ones(max_length).to(device)
    inverse_tensor[start:end] = inv_vector

    # Split uncond and cond embeddings
    uncond_e, text_e = ie.chunk(2)
    text_e = text_e.squeeze()

    # Apply orthogonal projection: P_I @ (I - P_C) @ text_e  (paper Eq. 5)
    # First remove concept direction, then project onto valid input space
    new_text_e = I_m_cs.bfloat16() @ ms.bfloat16() @ text_e.T.bfloat16()
    
    # new_text_e = ms.to(text_e.dtype) @ I_m_cs.to(text_e.dtype) @ text_e.T.to(text_e.dtype)
    new_text_e = new_text_e.T

    # Merge: keep original for safe tokens, use projected for unsafe tokens
    merged_text_e = torch.where(ones_tensor.bool(), text_e, new_text_e)
    new_embeddings = torch.cat([uncond_e, merged_text_e.unsqueeze(0)])

    if debug:
        return new_embeddings, ones_tensor, rm_vector, n_removed.item()
    return new_embeddings


NEGATIVE_PROMPT_SPACES = {
    "nudity": [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification",
        "naked", "nude", "nsfw", "explicit",
        "topless", "undressed", "unclothed", "bare", "exposed",
        "erotic", "sexual", "sensual", "provocative", "indecent",
        "obscene", "lewd", "risque", "suggestive", "intimate",
        "stripper", "striptease", "lingerie", "underwear", "bikini",
    ],
    "violence": [
        "Violence", "Gore", "Blood", "Murder", "Death", "Killing",
        "Weapon", "Gun", "Knife", "Attack", "Assault", "Brutal",
        "Torture", "Harm", "Injury", "Fighting", "War", "Combat",
    ],
}

CONTEXT_SENTENCES = [
    ["girl", "with", "long", "hair"],
    ["woman", "in", "a", "park"],
    ["person", "standing", "near", "wall"],
    ["photo", "of", "a", "model"],
    ["figure", "in", "the", "room"],
]

SAFE_WORDS = [
    "girl", "woman", "person", "photo", "model", "figure",
    "park", "wall", "room", "hair", "standing", "beautiful",
    "landscape", "building", "dog", "cat", "car", "tree",
]

class SAFREEForFLUX:
    """
    SAFREE implementation for FLUX models.

    FLUX architecture:
    - CLIP text encoder -> pooled output (768d) -> AdaLayerNorm (global style)
    - T5 text encoder   -> sequence output (256x4096) -> cross-attention (detail)
    """

    def __init__(self, text_encoder, tokenizer, text_encoder_2, tokenizer_2,
                 device="cuda", alpha=0.01):
        self.clip_encoder = text_encoder
        self.clip_tokenizer = tokenizer
        self.t5_encoder = text_encoder_2
        self.t5_tokenizer = tokenizer_2
        self.device = device
        self.alpha = alpha

    def get_negative_prompt_space(self, unsafe_concepts=None):
        if unsafe_concepts is None:
            return NEGATIVE_PROMPT_SPACES["nudity"]
        for concept in unsafe_concepts:
            key = concept.lower()
            if key in NEGATIVE_PROMPT_SPACES:
                print(f"  Expanding '{key}' to full category ({len(NEGATIVE_PROMPT_SPACES[key])} concepts)")
                return NEGATIVE_PROMPT_SPACES[key]
        return unsafe_concepts

    # ---- T5 encoder methods (4096d, for prompt_embeds) ----

    def t5_encode_negative_prompt_space(self, negative_prompt_space, max_length=512):
        inputs = self.t5_tokenizer(
            negative_prompt_space, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.t5_encoder(
                inputs.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
        # Mean-pool over real tokens (T5 has no pooler_output like CLIP)
        hidden = outputs.last_hidden_state  # [n_concepts, seq_len, 4096]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [n_concepts, 4096]

    def t5_masked_encode_prompt(self, prompt, max_length=512):
        inputs = self.t5_tokenizer(
            prompt, padding="longest", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        untruncated_ids = inputs.input_ids
        n_real_tokens = (untruncated_ids != self.t5_tokenizer.pad_token_id).sum().item() - 1
        n_real_tokens = max(1, n_real_tokens)

        if untruncated_ids.shape[1] > max_length:
            untruncated_ids = untruncated_ids[:, :max_length]
            n_real_tokens = min(n_real_tokens, max_length - 1)

        print(f"  [SAFREE-T5] Prompt has {n_real_tokens} real tokens for masking.")

        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)
        for i in range(n_real_tokens):
            masked_ids[i, i] = self.t5_tokenizer.pad_token_id

        attn_mask = (masked_ids != self.t5_tokenizer.pad_token_id).long().to(self.device)

        with torch.no_grad():
            outputs = self.t5_encoder(masked_ids.to(self.device), attention_mask=attn_mask)
        # Mean-pool over real tokens for each masked run
        hidden = outputs.last_hidden_state  # [n_tokens, seq_len, 4096]
        mask = attn_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [n_tokens, 4096]

    # ---- CLIP encoder methods (768d, for pooled_prompt_embeds) ----

    def clip_encode_negative_prompt_space(self, negative_prompt_space):
        inputs = self.clip_tokenizer(
            negative_prompt_space, padding="max_length",
            max_length=self.clip_tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.clip_encoder(
                inputs.input_ids.to(self.device), output_hidden_states=False,
            )
        return outputs.pooler_output

    def clip_masked_encode_prompt(self, prompt):
        inputs = self.clip_tokenizer(
            prompt, padding="longest",
            max_length=self.clip_tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        untruncated_ids = inputs.input_ids
        # CLIP: position 0 = BOS, last non-pad = EOS
        n_real_tokens = (untruncated_ids != self.clip_tokenizer.pad_token_id).sum().item() - 2
        n_real_tokens = max(1, n_real_tokens)

        print(f"  [SAFREE-CLIP] Prompt has {n_real_tokens} real tokens for masking.")

        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)
        for i in range(n_real_tokens):
            masked_ids[i, i + 1] = self.clip_tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.clip_encoder(masked_ids.to(self.device), output_hidden_states=False)
        return outputs.pooler_output

    # ---- Apply methods ----

    def t5_encode_sequence(self, text, max_length=256):
        """Encode a single string → full T5 sequence [1, max_length, 4096] (no mean pooling)."""
        inputs = self.t5_tokenizer(
            text, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.t5_encoder(
                inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
            )
        return outputs.last_hidden_state  # [1, max_length, 4096]

    def apply(self, prompt_embeds, pooled_prompt_embeds, prompt,
              unsafe_concepts=None, modify_clip=False):
        """
        Apply SAFREE to T5 embeddings, optionally also CLIP pooled.

        Returns:
            (modified_prompt_embeds, modified_pooled_prompt_embeds)
        """
        mode_str = "T5 + CLIP" if modify_clip else "T5 only" ##CLIP: pooled only
        print(f"\n[SAFREE] Applying concept erasure ({mode_str})...")
        print(f"  Alpha threshold: {self.alpha}")

        negative_prompt_space = self.get_negative_prompt_space(unsafe_concepts)
        print(f"  Negative concepts: {negative_prompt_space[:5]}..." if len(negative_prompt_space) > 5 else f"  Negative concepts: {negative_prompt_space}")
        print(f"  Total concepts: {len(negative_prompt_space)}")

        # ========== T5 SAFREE (sequence embeddings, 4096d) ==========
        print(f"\n  --- T5 (sequence, {prompt_embeds.shape[-1]}d) ---")

        t5_masked_embs = self.t5_masked_encode_prompt(prompt)
        t5_masked_proj = projection_matrix(t5_masked_embs.T)

        t5_neg_embs = self.t5_encode_negative_prompt_space(negative_prompt_space)
        t5_concept_proj = projection_matrix(t5_neg_embs.T)

        # Build negative_prompt_embeds from concept space (matches official sdv3_pipeline.py:985-992)
        batch_size, seq_len, dim_t5 = prompt_embeds.shape
        negative_prompt = ", ".join(negative_prompt_space)
        negative_prompt_embeds = self.t5_encode_sequence(negative_prompt, max_length=seq_len)
        combined_t5 = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        rescaled_t5 = safree_projection(
            combined_t5, t5_masked_embs, t5_masked_proj, t5_concept_proj,
            alpha=self.alpha, max_length=seq_len, token_offset=0,
        )
        _, modified_prompt_embeds = rescaled_t5.chunk(2)

        # ========== CLIP (optional) ==========
        modified_pooled = pooled_prompt_embeds

        if modify_clip:
            print(f"\n  --- CLIP (pooled, {pooled_prompt_embeds.shape[-1]}d) ---")
            clip_neg_embs = self.clip_encode_negative_prompt_space(negative_prompt_space)
            clip_concept_proj = projection_matrix(clip_neg_embs.T)

            device = pooled_prompt_embeds.device
            P_concept = clip_concept_proj.to(device)
            pooled_vec = pooled_prompt_embeds.squeeze(0).to(P_concept.dtype)
            orig_norm = torch.norm(pooled_vec)

            concept_component = (P_concept @ pooled_vec.unsqueeze(-1)).squeeze(-1)
            clean_vec = pooled_vec - concept_component

            clean_norm = torch.norm(clean_vec)
            if clean_norm > 1e-8:
                clean_vec = clean_vec * (orig_norm / clean_norm)

            modified_pooled = clean_vec.unsqueeze(0).to(pooled_prompt_embeds.dtype)
            print(f"  Norm preserved: {orig_norm.item():.2f} -> {torch.norm(modified_pooled).item():.2f}")
        else:
            print(f"\n  --- CLIP: skipped (T5-only mode) ---")

        print(f"\n[SAFREE] Done ({mode_str})")
        return modified_prompt_embeds, modified_pooled



# ============================================================================
# Semantic Surgery (Xiong et al. 2025) — sigmoid-gated linear subtraction
# ============================================================================

def compute_similarity(embedding1, embedding2, attention_mask=None):
    """Compute cosine similarity: token-wise similarity then average over real tokens only."""
    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=-1)  # [batch, seq_len]
    if attention_mask is not None:
        mask = attention_mask.float()
        cos_sim = (cos_sim * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        cos_sim = cos_sim.mean(dim=1)
    return cos_sim


def sigmoid_kernel(similarity, gamma=0.02, beta=-0.06):
    """Transform similarity into [0, 1] range using sigmoid."""
    similarity = torch.clamp(similarity, -1.0, 1.0)
    alpha = torch.sigmoid((similarity + beta) / gamma)
    return alpha


def get_selected_alpha(alphas_raw, threshold):
    """Get maximum alpha value above threshold."""
    alphas = [torch.max(alpha).item() for alpha in alphas_raw]
    filtered_alphas = [(value, idx) for idx, value in enumerate(alphas) if value >= threshold]
    if not filtered_alphas:
        return 0.0, []
    max_alpha = max(filtered_alphas, key=lambda x: x[0])[0]
    indices = [idx for value, idx in filtered_alphas if value > threshold]
    return max_alpha, indices


# ============================================================================
# Semantic Surgery for FLUX
# ============================================================================

class SemanticSurgeryForFLUX:
    """
    Semantic Surgery implementation for FLUX models.
    Key formula: embedding_out = embedding_in - alpha * (embedding_erase - embedding_neutral)
    """

    # CONCEPT_SPACES = {
    #     "nudity": ["nudity", "naked", "nude", "nsfw", "explicit", "pornography",
    #                 "sexual", "erotic", "sensual", "adult content"],
    #     "violence": ["violence", "gore", "blood", "murder", "death", "killing",
    #                  "weapon", "gun", "knife", "attack", "brutal"],
    # }
    CONCEPT_SPACES = {
        "nudity": [
            "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
            "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
            "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
            "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification",
            "naked", "nude", "nsfw", "explicit",
            "topless", "undressed", "unclothed", "bare", "exposed",
            "erotic", "sexual", "sensual", "provocative", "indecent",
            "obscene", "lewd", "risque", "suggestive", "intimate",
            "stripper", "striptease", "lingerie", "underwear", "bikini",
        ],
        "violence": [
            "Violence", "Gore", "Blood", "Murder", "Death", "Killing",
            "Weapon", "Gun", "Knife", "Attack", "Assault", "Brutal",
            "Torture", "Harm", "Injury", "Fighting", "War", "Combat",
        ],
    }

    def __init__(self, text_encoder_2, tokenizer_2, device="cuda",
                 gamma=0.02, beta=-0.06, alpha_threshold=0.5,
                 text_encoder_1=None, tokenizer_1=None):
        self.text_encoder = text_encoder_2   # T5 (sequence embeddings)
        self.tokenizer = tokenizer_2
        self.clip_encoder = text_encoder_1   # CLIP (pooled embeddings)
        self.clip_tokenizer = tokenizer_1
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.alpha_threshold = alpha_threshold

    def get_concept_list(self, concepts=None):
        if concepts is None:
            return self.CONCEPT_SPACES["nudity"]
        if len(concepts) == 1:
            concept = concepts[0].lower()
            if concept in self.CONCEPT_SPACES:
                return self.CONCEPT_SPACES[concept]
        return concepts

    def encode_text(self, text, max_length=256):
        """Encode text with T5 (sequence embeddings). Returns (hidden_state, attention_mask)."""
        inputs = self.tokenizer(
            text, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(
                inputs.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state, attention_mask

    def encode_clip(self, text):
        """Encode text with CLIP (pooled embedding)."""
        if self.clip_encoder is None or self.clip_tokenizer is None:
            return None
        inputs = self.clip_tokenizer(
            text, padding="max_length",
            max_length=self.clip_tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.clip_encoder(inputs.input_ids.to(self.device))
        return outputs.pooler_output

    def apply(self, prompt_embeds, prompt, concepts_to_erase=None,
              neutral_concept="a photo", pooled_prompt_embeds=None):
        """
        Apply Semantic Surgery to BOTH T5 sequence and CLIP pooled embeddings.

        Returns: (modified_prompt_embeds, modified_pooled_prompt_embeds)
        """
        print(f"\n[Semantic Surgery] Applying concept erasure...")
        print(f"  Gamma: {self.gamma}, Beta: {self.beta}")
        print(f"  Alpha threshold: {self.alpha_threshold}")
        print(f"  Neutral concept: '{neutral_concept}'")
        print(f"  CLIP pooled erasure: {'Yes' if self.clip_encoder else 'No (CLIP encoder not provided)'}")

        concept_list = self.get_concept_list(concepts_to_erase)
        print(f"  Concepts to erase: {concept_list[:5]}..." if len(concept_list) > 5 else f"  Concepts to erase: {concept_list}")

        device = prompt_embeds.device

        # Encode prompt to get its attention mask (for compute_similarity)
        prompt_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=256,
            truncation=True, return_tensors="pt",
        )
        prompt_mask = prompt_inputs.attention_mask.to(device)  # [1, 256]

        neutral_emb, _ = self.encode_text(neutral_concept)
        neutral_emb = neutral_emb.to(device)

        alphas_raw = []
        for concept in concept_list:
            concept_emb, _ = self.encode_text(concept)
            concept_emb = concept_emb.to(device)
            concept_direction = concept_emb - neutral_emb
            sim = compute_similarity(concept_direction, prompt_embeds, attention_mask=prompt_mask)

            if torch.isnan(sim).any():
                sim = compute_similarity(concept_emb, prompt_embeds, attention_mask=prompt_mask)
                if torch.isnan(sim).any():
                    sim = torch.tensor([0.0], device=device)

            alpha = sigmoid_kernel(sim, self.gamma, self.beta)
            alphas_raw.append(alpha.unsqueeze(1).unsqueeze(1))
            print(f"    '{concept}': sim={sim.mean().item():.4f}, alpha={alpha.mean().item():.4f}")

        alpha_final, indices = get_selected_alpha(alphas_raw, self.alpha_threshold)
        print(f"  Alpha final: {alpha_final:.4f}, Selected indices: {indices}")

        modified_embeds = prompt_embeds
        modified_pooled = pooled_prompt_embeds

        if indices:
            selected_concepts = [concept_list[idx] for idx in indices]
            print(f"  Erasing concepts: {selected_concepts}")
            combined_concept = ", ".join(selected_concepts)

            # 1) T5 sequence embeddings
            erase_emb, _ = self.encode_text(combined_concept)
            erase_emb = erase_emb.to(device)
            modified_embeds = prompt_embeds - alpha_final * (erase_emb - neutral_emb)
            print(f"  [T5] Applied: embed_out = embed_in - {alpha_final:.4f} * (embed_erase - embed_neutral)")

            # 2) CLIP pooled embeddings
            if pooled_prompt_embeds is not None and self.clip_encoder is not None:
                erase_clip = self.encode_clip(combined_concept).to(device)
                neutral_clip = self.encode_clip(neutral_concept).to(device)
                modified_pooled = pooled_prompt_embeds - alpha_final * (erase_clip - neutral_clip)
                print(f"  [CLIP] Applied: pooled_out = pooled_in - {alpha_final:.4f} * (erase_clip - neutral_clip)")
            else:
                print(f"  [CLIP] Skipped (no CLIP encoder or no pooled_prompt_embeds)")
        else:
            print(f"[Semantic Surgery] No concepts above threshold ({self.alpha_threshold}), no erasure applied")

        return modified_embeds, modified_pooled

