<!-- scope: code-specialized Evol-Instruct applied to HumanEval/MBPP-style coding instructions
     deps: [[evol-instruct]]
     see-also: [[oss-instruct]], [[code-evol-instruct]], [[wizardmath]]
-->

# WizardCoder: Empowering Code LLMs with Code Evol-Instruct
- **Core Insight:** Applying Evol-Instruct's depth/breadth mutations with code-specific operators (add constraints, require specific languages/libraries, increase time/space constraints, introduce edge cases) turns Code-Alpaca-scale data (~20K seeds) into a ~78K-sample set that pushes 15B open code models past Claude and Bard on HumanEval/HumanEval+.
- **Guideline:** For code SFT, start from a small seed (e.g. Code-Alpaca 20K), apply five or six code-specialized evolution operators per seed, run teacher generation, drop duplicates + stale / identical responses; ~4× expansion is enough.
- **Authors:** Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang (Microsoft)
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2306.08568
- **Relevant topics:** code SFT, Code Evol-Instruct, WizardCoder

## Abstract
WizardCoder is the first explicit port of the Evol-Instruct framework to code. It defines code-specific evolution operators — adding constraints, increasing time/space complexity, requiring specific libraries or languages, introducing misleading requirements, inserting edge cases — and runs them over Code-Alpaca seeds to produce ~78K high-complexity coding instructions. Fine-tuned WizardCoder-15B (on StarCoder base) beats Anthropic's Claude and Google's Bard on HumanEval and HumanEval+, and closely approaches GPT-3.5 on HumanEval. It established Code Evol-Instruct as a standard lever for code SFT data.

## Key Contributions
- Five code-specialized Evol-Instruct mutation operators (concrete operators enumerated below).
- Empirical demonstration that code evolution > code Self-Instruct at matched compute.
- Public WizardCoder-15B / -34B / -Mistral-7B checkpoints.
- Set up the code-SFT data framework later refined by [[oss-instruct]] (Magicoder) and [[code-evol-instruct]].

## Key Figures/Tables to Study
- **Table 1** — HumanEval / HumanEval+ / MBPP / DS-1000 / MultiPL-E across WizardCoder sizes vs closed/open baselines.
- **Operator examples figure** — seed problem + its five evolved variants.
- **Scaling curve** — data size vs HumanEval pass@1.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:** Code-Alpaca 20K seeds (itself Self-Instruct-generated on coding tasks).

- **Generation step(s) — five evolution operators:**
  1. **Add new constraints / requirements** (e.g., "the function must also handle negative integers").
  2. **Replace a common requirement with a less common one** (e.g., use `deque` instead of `list`).
  3. **Increase depth / reasoning steps** (multi-step algorithmic transformation).
  4. **Deepen problem complexity** (higher time/space constraint, more edge cases, misleading wordings).
  5. **Require specific language or library** (e.g., port to Rust, use NumPy vectorization).

  Per seed, one operator is randomly chosen; teacher re-generates instruction + reference solution.

- **Filtering/rescoring:**
  - Drop evolutions where teacher refuses or the new instruction is identical to the old.
  - Length-based sanity filter.
  - Deduplicate via exact/near-exact match.
  - Decontamination vs HumanEval / MBPP.

- **Output shape:** ~78K evolved `<instruction, response>` coding pairs released under the WizardLM family license.

- **Teacher model(s):** GPT-3.5-Turbo (at release); later iterations use stronger teachers.

- **Cost estimate:** not fully disclosed; O(one GPT-3.5 call per evolution).

## Quality / diversity evaluation
- WizardCoder-15B: HumanEval 57.3 / HumanEval+ 50.6 / MBPP 51.8 (beats Claude, Bard).
- WizardCoder-34B / WizardCoder-Python-34B: near GPT-3.5 on HumanEval.
- WizardCoder-Mistral-7B: best open 7B at release.
- Ablation: evolved data > raw Code-Alpaca at matched 20K and scales well with more evolutions.

## Risks + gotchas
- **Mode collapse risk** — repeated use of the same operator on the same seed yields near-duplicate outputs; operator randomization + filtering are load-bearing.
- **HumanEval contamination** — WizardCoder is one of the earliest cases where independent audits found stylistic overlap with HumanEval tasks; later authors recommend stricter decontamination.
- **License** on WizardLM family has fluctuated (removed/restored); check before redistribution.
- **Python-centric** — non-Python coverage is shallower.

## Connections
- Direct application of [[evol-instruct]] to code.
- Sibling to [[wizardmath]] (math-specialized evolution).
- Complementary to [[oss-instruct]] (snippet-grounded synthesis); MagicoderS combines both.
- Operator list is reused in more recent [[code-evol-instruct]]-style pipelines and in [[nemotron-4-synthetic]]'s code-category prompt synthesis.
