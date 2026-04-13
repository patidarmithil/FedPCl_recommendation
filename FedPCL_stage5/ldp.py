"""
ldp.py
══════
Local Differential Privacy (LDP) module for FedPCL Stage 5.

Implements paper Eq. (9):
  g̃u = clip(gu, σ) + Laplacian(0, λ)

where:
  clip(x, σ)  — clips the L2-norm of x to σ  (not element-wise clipping)
  Laplacian(0, λ) — zero-mean Laplace noise with scale λ

── Privacy accounting ────────────────────────────────────────────────────────
For Laplace mechanism, ε-DP holds when:
  λ = Δf / ε    where Δf = σ (the clipping threshold / sensitivity)
  → ε = σ / λ

We expose `epsilon` as the effective privacy budget per gradient upload.
Over T rounds: total epsilon ≤ T * epsilon_per_round  (basic composition).

── Design decisions ──────────────────────────────────────────────────────────
1. We clip the L2-norm of the WHOLE delta tensor (not per-element), matching
   the federated DP literature (McMahan et al. 2018, NeurIPS).
2. Laplace noise is added element-wise after clipping.
3. The same function is applied to both item deltas and user embeddings.

Paper hyperparameters (Section IV-A):
  σ (clip threshold) : 1.0  — we use this as the default sensitivity bound
  λ (noise scale)    : 0.01 — small noise; paper does not specify exact value
  These give ε = σ/λ = 100 per round (relatively weak LDP for performance).
  Tighten λ (e.g. 0.1) for stronger privacy at the cost of accuracy.
"""
# added some additional comment
import torch


# ══════════════════════════════════════════════════════════════════════════════
def clip_tensor(t: torch.Tensor, clip_norm: float) -> torch.Tensor:
    """
    Clip a tensor so its L2-norm does not exceed `clip_norm`.

    g_clipped = g * min(1, clip_norm / ||g||_2)

    Args:
        t:         arbitrary-shape tensor
        clip_norm: maximum L2 norm allowed

    Returns:
        clipped tensor (same shape, same device)
    """
    norm = t.norm(p=2)
    if norm > clip_norm:
        t = t * (clip_norm / norm)
    return t


# ══════════════════════════════════════════════════════════════════════════════
def laplace_noise(shape, scale: float, device: torch.device) -> torch.Tensor:
    """
    Sample element-wise Laplace noise ~ Laplace(0, scale).

    Uses PyTorch's built-in Laplace distribution which is numerically safe.

    Previous manual inverse-CDF approach had a NaN/inf bug:
      u = Uniform(-0.5, 0.5)
      noise = -scale * sign(u) * log1p(-2|u|)
    When u is exactly ±0.5 (possible in float32), log1p(-1) = log(0) = -inf,
    producing ±inf noise that corrupts embeddings.

    Args:
        shape:  tuple of ints
        scale:  Laplace scale parameter λ
        device: torch device

    Returns:
        noise tensor [*shape]
    """
    dist  = torch.distributions.Laplace(
        loc   = torch.zeros(shape, device=device),
        scale = torch.full(shape, scale, device=device),
    )
    return dist.sample()


# ══════════════════════════════════════════════════════════════════════════════
def apply_ldp(t: torch.Tensor,
              clip_norm: float = 1.0,
              noise_scale: float = 0.01) -> torch.Tensor:
    """
    Apply LDP to a single tensor: clip then add Laplace noise.

    Implements paper Eq. (9):
      g̃ = clip(g, σ) + Laplacian(0, λ)

    Args:
        t:           tensor to protect (gradient or delta)
        clip_norm:   σ — clipping threshold
        noise_scale: λ — Laplace noise scale

    Returns:
        privatised tensor (same shape/device as t)
    """
    t_clipped = clip_tensor(t.detach(), clip_norm)
    noise     = laplace_noise(t.shape, noise_scale, t.device)
    return t_clipped + noise


# ══════════════════════════════════════════════════════════════════════════════
def apply_ldp_to_deltas(item_deltas: dict,
                        clip_norm: float = 1.0,
                        noise_scale: float = 0.01) -> dict:
    """
    Apply LDP to a dict of item deltas {item_id: delta_tensor}.

    Each delta is clipped independently (per-item sensitivity).
    Noise is added per delta tensor.

    Args:
        item_deltas: {iid: [d] tensor}
        clip_norm:   σ
        noise_scale: λ

    Returns:
        {iid: privatised [d] tensor}
    """
    return {
        iid: apply_ldp(delta, clip_norm, noise_scale)
        for iid, delta in item_deltas.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
def privacy_budget(clip_norm: float, noise_scale: float) -> float:
    """
    Compute the per-round ε for the Laplace mechanism.

    ε = sensitivity / λ = σ / λ

    Args:
        clip_norm:   σ (clipping threshold = L2 sensitivity)
        noise_scale: λ (Laplace scale)

    Returns:
        ε (epsilon) per upload
    """
    return clip_norm / noise_scale if noise_scale > 0 else float('inf')
