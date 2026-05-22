<!-- scope: layer-skipping self-speculative decoding without auxiliary models
     deps: fast-inference-from-transformers-via-speculative-decoding
     see-also: lookahead-decoding, medusa
-->

# Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding
- **Core Insight:** The same LLM can draft cheaply by skipping selected layers, then verify with the full model in one pass.
- **Guideline:** Use self-speculative decoding when extra model memory is unacceptable and the target architecture tolerates lower-quality layer-skipped drafts.
- **Authors:** Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Sharad Mehrotra
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.08168
- **Relevant topics:** self-speculative decoding, layer skipping, lossless acceleration, no auxiliary model, draft and verify

## Abstract
Draft & Verify proposes self-speculative decoding for accelerating LLM inference without an auxiliary draft model. During drafting, the model skips selected intermediate layers to produce tokens faster but with lower quality. During verification, the full original model validates the drafted tokens in one forward pass, preserving the original model's output. The paper reports speedups up to 1.99x on LLaMA-2 variants.

## Key Contributions
- Removes the need for a separate assistant model and its memory footprint.
- Uses layer skipping as the cheap draft path inside the target model.
- Verifies draft tokens with the full unmodified model.
- Requires no additional neural network training.
- Studies skip-layer choices and speed/acceptance tradeoffs.

## Key Figures/Tables to Study
- Drafting/verification pipeline figure: shows the same model in two modes.
- Layer skipping analysis: which layers can be skipped with acceptable draft quality.
- Speedup tables on LLaMA-2: inspect accepted tokens and wall-clock gains.
- Correctness discussion: how full-model verification preserves output.

## Technical Details
The draft phase runs a shallower effective network by skipping intermediate layers and autoregressively producing several candidate tokens. The verify phase feeds the draft sequence through the complete model and accepts the longest prefix that matches the target model's decisions under the supported decoding mode.

The main systems benefit is memory: no assistant model and no extra heads are required. The main risk is acceptance rate. If skipped-layer drafts diverge often, the saved draft cost may not compensate for verification overhead.

## Connections
- [[lookahead-decoding]] is another no-assistant exact acceleration method.
- [[medusa]] adds heads instead of skipping layers.
- [[hf-assisted-generation]] covers the standard assistant-model path, which may get higher acceptance but costs extra memory.
- [[eagle]] uses a trained feature drafter, occupying the middle ground between assistant models and layer skipping.
- [[lookahead-decoding]] avoids both auxiliary weights and explicit layer skipping.
