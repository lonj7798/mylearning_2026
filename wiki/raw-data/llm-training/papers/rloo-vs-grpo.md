<!-- scope: pointer card — where the RLOO/GRPO comparison claims actually come from; no primary artifact of its own
     deps: [[rloo]], [[grpo]]
     see-also: [[dr-grpo]], [[reinforce-plus-plus]], [[ppo]]
-->

# RLOO vs GRPO — pointer card (no primary source)

> **No verifiable primary source.** No paper, technical report, official blog post, or repository
> titled "RLOO vs GRPO" or presenting this comparison as one artifact was found. Searched on
> 2026-09-18: arXiv listings for Ahmadian et al. (2402.14740), Shao et al. (2402.03300), Liu et al.
> (2503.20783), Hu et al. (REINFORCE++), the TRL and OpenRLHF documentation and repositories,
> and the Interconnects post cached as [[nathan-lambert-grpo]]. The previous version of this card
> listed its author field as "(synthesized from Ahmadian 2024, Shao 2024, Liu 2025, Hu 2025, plus
> practitioner blog posts from HF and OpenRLHF)" and gave no URL, which is a synthesis rather than
> an extract of one artifact.

Chapters must not cite this card as evidence. Cite the card for the algorithm whose property is
being claimed. The pointers below exist so that existing wikilinks resolve and so that each claim
that used to live here can be traced to the source that actually makes it.

## Where each comparison claim lives

| Claim | Primary source | Locus (verified 2026-09-18) |
|---|---|---|
| RLOO baseline is the mean reward of the other k − 1 samples; unbiased; no value network | [[rloo]] — arXiv:2402.14740v2 | §2.3, unnumbered display |
| RLOO experiments use k = 2 and k = 4 only | [[rloo]] | §5.1, Table 1; App. C |
| RLOO applies the KL penalty as a per-token shaped reward | [[rloo]] | §2.2, Eq. 3 and 4 |
| GRPO advantage is `(r_i − mean(r)) / std(r)`, assigned to every token of output `o_i` | [[grpo]] — DeepSeekMath, arXiv:2402.03300 | §4.1.2 |
| GRPO adds the KL divergence directly to the loss, using the unbiased (k3) estimator, not to the reward | [[grpo]] | §4.1.1, Eq. 4 |
| GRPO's released run: G = 64 outputs per question, KL coefficient 0.04, policy LR 1e-6, batch 1024, max length 1024, a single policy update per exploration stage | [[grpo]] | §4.2 |
| GRPO's clip parameter ε and iteration count μ are hyperparameters of Algorithm 1; no numeric ε is printed in the paper | [[grpo]] | §4.1.1, Algorithm 1 |
| Dividing by `std(R)` up-weights questions whose rewards are almost all 1 or almost all 0 — "question-level difficulty bias" | [[dr-grpo]] — arXiv:2503.20783 | §3.1 and Figure 4 |
| Dividing by `|o_i|` up-weights short correct responses and under-penalizes long incorrect ones — "response-level length bias" | [[dr-grpo]] | §3.1 and Figure 4 |
| Dr. GRPO removes both the std and the length normalization terms | [[dr-grpo]] | §3.1, Figure 1 |
| Prompt-level normalization (GRPO, RLOO) is a mathematically biased estimator because the centered reward and the local standard deviation are not independent | [[reinforce-plus-plus]] | Abstract and App. A |
| REINFORCE++ normalizes advantages across the entire global batch instead of per prompt, with two variants (k ≥ 1 and k > 1 with a group baseline) | [[reinforce-plus-plus]] | Abstract, §1 |

## Claims that were on this card and are not supported anywhere checked

- "Their advantages become equivalent (up to scaling by std and epsilon clipping) as the number of
  samples per prompt grows." No source checked states this equivalence. [[rloo]] does not discuss
  GRPO; [[grpo]] does not discuss RLOO. Treat any such statement as an Interpretation belonging to
  the chapter that makes it, not to a source card.
- "RLOO ≈ GRPO without std normalization and without clip ≈ Dr. GRPO." Same status. [[dr-grpo]]
  derives its objective from GRPO by removing two terms (§3.1); it does not claim identity with RLOO.
- "GRPO uses ε = 0.2." Not printed in [[grpo]]. 0.2 is a framework default, recorded in the framework
  cards ([[verl-grpo]], [[trl-grpo]]), and a framework default is a separate fact from a paper value.
- "Epochs per rollout: 1 for both." [[grpo]] §4.2 states a single update per exploration stage for
  DeepSeekMath-RL; [[rloo]] App. C states 2 gradient steps per batch. These are different.
- The "When each wins empirically" table and the four numbered "practitioner recommendations". No
  head-to-head experiment comparing RLOO and GRPO under matched conditions was found in any source
  checked, and the HF, OpenRLHF, and verl documentation cited for them does not contain such a
  ranking.
- "std(R) degenerates when a group is all-right or all-wrong." The verifiable statement is narrower:
  [[dr-grpo]] §3.1 states that low-std questions receive *higher* weight in the update, which is a
  weighting bias, not a division-by-zero failure. Implementations add an epsilon to the denominator;
  check the framework card for the value used.

## Connections
- Component algorithms: [[rloo]], [[grpo]], [[dr-grpo]], [[reinforce-plus-plus]].
- Shared ancestor: [[vanilla-pg]].
- The method all four replace: [[ppo]].
- Framework implementations, where default values such as clip ε live: [[verl-grpo]], [[trl-grpo]], [[openrlhf-ppo]].
- Practitioner commentary: [[nathan-lambert-grpo]].

## Verification
- Checked on 2026-09-18. No primary artifact corresponds to this slug and title; the card has been
  reduced to pointers under the "No verifiable primary source" rule.
- Corrections to the previous card version: the card presented itself as a source extract with a
  Core Insight, Guideline, author list, and year range while citing no artifact; every factual row it
  carried is now attributed to the card and locus that actually supports it, and the rows that no
  source supports are listed above instead of being stated as fact.
- Removed as unsupported by any source checked: the RLOO/GRPO equivalence-in-the-limit argument; the
  "RLOO ≈ Dr. GRPO" chain; ε = 0.2 attributed to GRPO the paper; the "When each wins empirically"
  table; the four practitioner recommendations attributed to HF, OpenRLHF, and verl documentation;
  the claim that GRPO causes "length inflation" attributed to no locus (the length bias is stated in
  [[dr-grpo]] §3.1 and should be cited there).
- Not reported by any source checked: a controlled head-to-head comparison of RLOO and GRPO on the
  same model, data, and reward.
