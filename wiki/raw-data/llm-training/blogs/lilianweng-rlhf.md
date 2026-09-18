<!-- scope: redirect card — no Lil'Log post on RLHF exists; see [[lilianweng-reward-hacking]]
     see-also: [[lilianweng-reward-hacking]], [[hf-rlhf-illustrated]], [[rlhf-instructgpt]], [[nathan-lambert-rl-overview]]
-->

# Redirect: lilianweng-rlhf → [[lilianweng-reward-hacking]]

> **No verifiable primary source.** This card described a Lil'Log post, or series of posts, on the RLHF
> pipeline. No such post exists. The card's URL, https://lilianweng.github.io/tags/rlhf/, is a tag index,
> and as of 2026-09-18 that tag lists exactly one post: "Reward Hacking in Reinforcement Learning"
> (2024-11-28), which is already covered by [[lilianweng-reward-hacking]]. The full Lil'Log archive
> (https://lilianweng.github.io/archives/, 52 posts checked on 2026-09-18) contains no post on
> reinforcement learning from human feedback. The card's title also named "The Transformer Family",
> a real Lil'Log post (2020-04-07) about transformer architecture variants, not about RLHF.

- **Status:** redirect stub. Do not cite this card as evidence.
- **Checked on:** 2026-09-18
- **Searched:** https://lilianweng.github.io/tags/rlhf/ (1 post), https://lilianweng.github.io/archives/ (full post list), https://lilianweng.github.io/posts/2024-11-28-reward-hacking/ (full text).

## Where the content actually lives
- [[lilianweng-reward-hacking]] — the one Lil'Log post under the `rlhf` tag. Covers the three rewards in
  an RLHF setup (oracle, human, proxy), reward-model overoptimization, judge bias, and in-context reward
  hacking. This is the card to link when a chapter wants Weng on RLHF.
- [[rlhf-instructgpt]] — Ouyang et al. 2022, the primary source for the three-stage SFT → reward model →
  PPO pipeline and for the KL-penalized objective.
- [[hf-rlhf-illustrated]] — tutorial-level walkthrough of the same pipeline with the Bradley-Terry
  reward-model loss.
- [[nathan-lambert-rl-overview]] — practitioner overview of RLHF algorithm choices.
- [[costa-huang-ppo-details]] — the implementation-level facts
  (reward whitening, advantage normalization, value-head sharing, per-token KL) that this card had
  attributed to Weng.
- [[ppo]], [[trpo]] — GAE, γ and λ, the clipped surrogate objective, and the entropy bonus coefficient.
- [[kl-control-rlhf]], [[john-schulman-kl-tricks]] — KL estimators and where the KL term is applied.

## Verification
- Checked on 2026-09-18 against: https://lilianweng.github.io/tags/rlhf/ and https://lilianweng.github.io/archives/
- Corrections to the previous card version: the card described an artifact that does not exist. Title
  "Lil'Log — 'The Transformer Family' lineage: RLHF posts" mixed a real post title with a topic that post
  does not cover; **Year** "2023 series" is unsupported (the only `rlhf`-tagged post is from 2024-11);
  **URL** pointed at a tag index rather than an artifact.
- Removed as unsupported by the source: the Bradley-Terry derivation, the per-token reward form
  `r_total = r(x,y)·1[y=EOS] − β·log(π/π_ref)`, reward whitening, GAE λ = 0.95 with γ = 1.0, the
  value-head sharing tradeoff, the claim that the entropy bonus is usually dropped in RLHF, the
  three-stage pipeline diagram, and the reward-hacking taxonomy of "length hacking, sycophancy,
  specification gaming". None of these appear in any Lil'Log post attributable to this slug. Several are
  true statements about RLHF, but their support is [[ppo]], [[rlhf-instructgpt]],
  [[costa-huang-ppo-details]], not Weng.
- **Chapters to repair:** `wiki/courses/llm-training/ch-37/read_kor.md` and
  `wiki/courses/llm-training/ch-38/read.md` / `read_kor.md` cite this slug as evidence for the entropy-bonus,
  γ = 1.0, λ = 0.95, and per-token KL claims, including a direct quotation attributed to Weng. Those
  citations need to be repointed to the cards listed above. `wiki/raw-data/llm-training/blogs/hf-rlhf-illustrated.md`,
  `nathan-lambert-rl-overview.md`, and `lilianweng-reasoning-llms.md` link to this slug in "see-also" and
  Connections lines only, which this redirect resolves.
