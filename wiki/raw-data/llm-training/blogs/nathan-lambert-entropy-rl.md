<!-- scope: redirect / no-primary-source notice; entropy in LLM RL is covered by [[entropy-mechanism-llm-rl]]
     see-also: [[entropy-mechanism-llm-rl]], [[nathan-lambert-grpo]], [[rlvr-tulu3]]
-->

# Nathan Lambert (Interconnects) on Entropy in LLM RL

> **No verifiable primary source.** No Interconnects post devoted to entropy in RL for language models could be
> located on 2026-09-18. The previous version of this card was a synthesis across unnamed posts, with a homepage URL
> (https://www.interconnects.ai/) instead of an article URL, and it carried quotations and numeric thresholds that
> could not be traced to any published text. Under §9 of the authoring standard a card must extract one primary
> artifact, so the body has been replaced by this notice. Chapters must not cite this card.

## What was searched (2026-09-18)
- The Interconnects archive search page for "entropy": no post about entropy in RL was returned.
- The Interconnects 2025 sitemap (https://www.interconnects.ai/sitemap/2025): no post title contains "entropy",
  "exploration", or "RLVR".
- The two post titles named by the previous card version were not found. The closest actual titles in the 2025 sitemap
  are "Quick recap on the state of reasoning" (https://www.interconnects.ai/p/the-state-of-reasoning) and
  "Where inference-time scaling pushes the market for AI companies"; neither is the claimed
  "The state of reasoning & inference-time compute" or "RLVR and verifiable rewards".

## Where the claims should be sourced instead
| Claim previously made here | Card to use instead |
|---|---|
| Entropy collapse mechanism; entropy as a monitored training signal | [[entropy-mechanism-llm-rl]] |
| Entropy bonus alone does not prevent collapse at LLM scale | [[entropy-mechanism-llm-rl]] |
| RLVR design and Tülu 3 RLVR hyper-parameters | [[rlvr-tulu3]] |
| GRPO variants, clip-higher and its effect on entropy, base-model RL | [[nathan-lambert-grpo]] |
| Reward-model over-optimization | [[reward-model-overoptimization]] |
| Reward hacking taxonomy | [[lilianweng-reward-hacking]] |

A verifiable Interconnects article by the same author, read in full and checked against its published text, is
[[nathan-lambert-grpo]] ("Recent reasoning research: GRPO tweaks, base model RL, and data curation", 2025-03-31). Use
that card when a practitioner-level Interconnects citation is wanted for reasoning RL.

## Verification
- Checked on 2026-09-18 against: https://www.interconnects.ai/archive (search "entropy") and
  https://www.interconnects.ai/sitemap/2025.
- Corrections to the previous card version: the card named no single artifact and gave a homepage URL; it is replaced
  by this notice rather than repaired, because there is no primary artifact matching the slug and title.
- Removed as unsupported by the source:
  - The quotation "finally, a reward that cannot be gamed" attributed to the author about Tülu 3 RLVR.
  - The threshold "when entropy crashes below ~0.2 nats on last-token distribution, stop and inspect", attributed to
    the author and to "OpenRLHF practitioner notes".
  - The claim that R1-Zero works because "GRPO + long rollouts + rule-based reward happen to sit in a low-entropy-
    collapse regime compared to PPO-RLHF".
  - The practitioner advice "if a run is flatlining, raise rollout temperature before retuning β".
  - The list of "monitored metrics in Tülu-3-style RLVR runs" presented as the author's; no locus was given, and
    Tülu 3's own logged metrics belong in [[rlvr-tulu3]].
  - The stated opinion that "DPO's implicit KL regularization partly protects against entropy collapse".
  - The four "Key Posts to Read" entries, two of whose titles do not exist in the Interconnects archive.
  - The framing that "every recent reasoning-RL paper either monitors entropy or invents a trick to keep it from
    crashing", which no located source states.
- Not reported: nothing is claimed by this card, so no fields are outstanding.
