# Excerpt: The RoPE Decoupling Problem in MLA

<!-- source: [[deepseek-v2|report]], Section 2.1.2 -->

## The Conflict

MLA compresses keys and values into a latent vector $c_t$ via a learned down-projection:

$$c_t = W_\text{DKV} \, x_t \qquad c_t \in \mathbb{R}^{512}$$

Keys are then reconstructed per-head from $c_t$:

$$k_t^{(h)} = W_{UK}^{(h)} \, c_t$$

The problem: **Rotary Position Embeddings (RoPE) must be applied to keys before they are used in attention.** RoPE encodes each key with a position-dependent rotation matrix $R_t$ that depends on the absolute position $t$:

$$k_t^\text{RoPE} = R_t \cdot k_t$$

If we apply RoPE to the full reconstructed key, we would need to store the RoPE-encoded key — which is full-dimensional ($H \times d_k = 16{,}384$ dims) — in the cache. That defeats the entire purpose of MLA compression.

If we apply RoPE to the latent $c_t$ before caching, the problem is that the up-projection $W_{UK}^{(h)}$ is a *linear* map, and RoPE rotation does **not commute** with arbitrary linear transformations. So $W_{UK}^{(h)} (R_t \cdot c_t) \neq R_t \cdot W_{UK}^{(h)} c_t$ in general. The positional information would be scrambled by the up-projection.

## The Solution: Decoupled Positional Keys

DeepSeek separates the key into two components:

1. **Content key** $k_t^{C,(h)} = W_{UK}^{(h)} c_t$ — reconstructed from the latent, carries no positional information. The latent $c_t$ is cached.

2. **Positional key** $k_t^{R} \in \mathbb{R}^{64}$ — a small, separate key computed directly from $x_t$ with RoPE applied. This is cached alongside $c_t$.

The attention score for head $h$ at query position $s$ attending to key position $t$ becomes:

$$\text{score}_{s,t}^{(h)} = \underbrace{q_s^{C,(h)\top} k_t^{C,(h)}}_\text{content matching} + \underbrace{q_s^{R\top} k_t^R}_\text{position matching}$$

The content component captures *what* information is at position $t$. The positional component captures *where* position $t$ is relative to position $s$. They are computed independently and summed.

## Cache Composition

Per token, per layer, the KV cache stores:

| Component | Dimensions | Purpose |
|-----------|-----------|---------|
| $c_t$ (KV latent) | 512 | Shared basis for K and V reconstruction |
| $k_t^R$ (RoPE key) | 64 | Positional information for attention |
| **Total** | **576** | vs. 32,768 for standard MHA |

The 64-dimensional RoPE key is shared across all heads — there is no per-head RoPE component. This works because relative position is the same for all heads; it's the content-dependent attention that needs per-head diversity, and that comes from the per-head up-projection of $c_t$.

## Why 64 Dimensions for RoPE?

The RoPE key dimension $d_h^R = 64$ is a design choice. DeepSeek-V2's ablations showed that 64 dimensions capture sufficient positional information for the model to maintain strong length generalization. RoPE encodes position through rotations of dimension pairs, so 64 dimensions provide 32 rotation frequencies — more than enough to distinguish positions within a 128K-token context.

Increasing $d_h^R$ would improve positional resolution but increase cache size. At 64 dims, the positional component adds only 12.5% to the cache (64/512), a negligible overhead for the positional capability it provides.

## Connection to [[ch-06]] and [[ch-07]]

The decoupled RoPE approach in MLA reflects a broader principle from [[ch-06]]: positional encoding and content representation serve different purposes and can be factored apart. MLA takes this factoring to its logical conclusion — content keys and positional keys are computed through entirely separate pathways, with different dimensionalities, and combined only at the point of attention score computation.

This contrasts with standard Transformer attention, where RoPE is "baked into" the key vector. That entanglement is why naive KV compression (e.g., simple dimensionality reduction of cached keys) destroys positional information. MLA's explicit factoring solves the problem architecturally rather than trying to preserve RoPE through the compression bottleneck.
