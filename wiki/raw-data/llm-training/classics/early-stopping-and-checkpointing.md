<!-- scope: validation-tracked early stopping, checkpoint contents, stochastic weight averaging, model souping, and checkpoint averaging as practised in LLM pretraining
     deps: [[lr-schedules]]
     see-also: [[adam]], [[dropout]], [[llama-3]], [[olmo-2]]
-->

# Early Stopping, Checkpointing, SWA, and Model Souping
- **Core Insight:** Averaging weights from several points of a training run, or from several runs started from the same checkpoint, gives a model that is at least as accurate as the best single point in the cases measured: SWA raises ImageNet top-1 by 0.6–0.9% over three pretrained architectures (SWA Table 2), a greedy soup raises CLIP ViT-B/32 ImageNet top-1 from 80.38 to 81.03 (Model Soups Table 3), and OLMo 2's three-way mid-training soup equals or beats the best single checkpoint on 6 of 6 mixes on OLMES, OLMES-Gen and MMLU (OLMo 2 Table 14).
- **Guideline:** When several checkpoints or runs share one initialization and one loss basin, average them and select by held-out validation score, because averaging costs nothing at inference and was no worse than the best single model in all cases above. When the runs come from different initializations or architectures, do not average them: the Model Soups result is specific to fine-tuning from a shared pre-trained initialization.
- **Card type:** composite classics card. It extracts four primary artifacts, each cited at its own locus:
  - Stochastic Weight Averaging — "Averaging Weights Leads to Wider Optima and Better Generalization", Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson, 2018 (arXiv v1 2018-03), https://arxiv.org/abs/1803.05407 — paper.
  - Model Soups — "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, et al., 2022 (arXiv v1 2022-03), https://arxiv.org/abs/2203.05482 — paper.
  - The Llama 3 Herd of Models, Meta, 2024, https://arxiv.org/abs/2407.21783 — official technical report ([[llama-3]]).
  - 2 OLMo 2 Furious, Ai2, 2025 (arXiv:2501.00656), https://arxiv.org/abs/2501.00656 — official technical report ([[olmo-2]]).
- **Relevant topics:** generalization, checkpointing, weight averaging, model merging, checkpoint selection

## Summary
Early stopping halts training when a held-out validation metric stops improving for a fixed number of evaluations, and returns the best-scoring checkpoint rather than the last one. Stochastic Weight Averaging (SWA) instead keeps training under a cyclical or high constant learning rate and returns an equally weighted average of the weights visited; the SWA paper reports that this finds wider optima and better test accuracy than the SGD endpoint at almost no extra cost. Model Soups applies weight averaging across separate fine-tuning runs from one shared pre-trained initialization, and selects which runs to include greedily on a held-out validation set. Two LLM technical reports use the same operation at pretraining scale: Llama 3 averages checkpoints during its annealing phase to produce the final pre-trained model, and OLMo 2 averages models annealed on different data orders to produce its released mid-trained models.

## Key Contributions
- **Early stopping**: checkpoint selection by a validation metric rather than by training-loss minimum or by final step.
- **SWA**: an equally weighted running average of the weights traversed by SGD under a cyclical or high constant learning rate, plus one extra pass over the training data to recompute BatchNorm statistics for the averaged weights (SWA §3.1).
- **Model Soups**: weight-space rather than output-space ensembling, with a greedy inclusion rule that keeps a candidate only if held-out validation accuracy improves (Model Soups Recipe 1), giving a soup that can be no worse than the best individual model on that validation set.
- **Checkpoint averaging in LLM pretraining**: Llama 3 uses Polyak averaging over annealing-phase checkpoints (Llama 3 §3.4.3) and averages models across data and hyperparameter variants at each of the RM, SFT and DPO stages (§4.1.5); OLMo 2 averages three or four separately annealed mid-training runs (OLMo 2 §4.5).

## Key Figures/Tables to Study
- **SWA Figure 1**: the loss-surface visualization used to argue that the SGD trajectory stays on the periphery of the high-performing region while its average moves inward.
- **SWA Table 2**: ImageNet top-1 for SGD versus SWA at 5 and 10 epochs, three architectures.
- **Model Soups Figure 1 and Table 1**: greedy soup versus the best individual model in a hyperparameter search, on ImageNet and on five distribution shifts.
- **OLMo 2 Table 14**: best single checkpoint versus three-checkpoint soup across six mid-training mixes.

## Technical Details

**Early stopping.** Evaluate on a held-out set every `eval_every` steps; track the best value; stop after `patience` consecutive evaluations without an improvement larger than `delta`; return the best checkpoint rather than the last one.
```
best_val = inf; patience_counter = 0; best_ckpt = None
for step in range(total_steps):
    train_step()
    if step % eval_every == 0:
        val = evaluate(val_set)
        if val < best_val - delta:
            best_val, best_ckpt, patience_counter = val, save_checkpoint(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
return best_ckpt
```
No specific `patience` or `delta` value is taken from a primary source here; both are run-level choices that must be reported with the run.

**SWA** (Izmailov et al. 2018). After a normal training phase, continue with a cyclical or high constant learning rate, snapshot the weights each cycle or each epoch, and maintain `w_swa ← w_swa + (w_new − w_swa) / (n + 1)`. The paper reports ImageNet top-1 (SGD → SWA at 10 epochs): ResNet-50 76.15 → 76.97 ± 0.05, ResNet-152 78.31 → 78.94 ± 0.07, DenseNet-161 77.65 → 78.44 ± 0.06, summarized as a consistent 0.6–0.9% improvement over the pretrained models (SWA §4.2, Table 2). It also reports training a Wide ResNet-28-10 on CIFAR-100 from random initialization with a fixed learning rate of 0.05 for 300 epochs, averaging weights from epoch 140 onward, reaching 81.7 test accuracy (SWA §4.4). If the network uses BatchNorm, one extra forward pass over the training data in training mode is required to collect activation statistics for the averaged weights, because those statistics are not collected during averaging (SWA §3.1). Transformer LLMs use RMSNorm or LayerNorm, which hold no running statistics, so this step does not apply to them.

**Model Soups** (Wortsman et al. 2022). Sort the fine-tuned models by held-out validation accuracy in decreasing order; add each in turn to the running average and keep it only if validation performance improves (Recipe 1). Reported results: fine-tuning a JFT-3B-pretrained ViT-G/14 on ImageNet, best model in the hyperparameter search 90.78 top-1 and 84.68 average over five distribution shifts, greedy soup 90.94 and 85.02 (Table 1); fine-tuning CLIP ViT-B/32, best individual model 80.38 ImageNet top-1 and 47.83 on distribution shifts, greedy soup 81.03 and 50.75, logit ensemble 81.19 and 50.77 (Table 3). The soups in this paper all start from one shared pre-trained initialization; the paper's framing is that such fine-tuned models "often appear to lie in a single low error basin" (Abstract).

**Checkpoint averaging in LLM pretraining.**
- Llama 3: during pre-training on the final 40M tokens the learning rate is annealed linearly to 0 at 128K context, the data mix is upsampled toward high-quality sources, and the final pre-trained model is the average of the checkpoints from that annealing phase, cited as Polyak averaging (Llama 3 §3.4.3). Separately, at each of the reward-model, SFT and DPO stages, models trained with different data versions or hyperparameters are averaged (§4.1.5). No delta in loss or benchmark score is reported for either operation.
- OLMo 2: for the 7B model, three separate 50B-token anneals with different randomized data orders are averaged; for the 13B and 32B models, three 100B-token anneals plus one 300B-token anneal are averaged, which the report says was empirically better than averaging the three 100B runs alone (OLMo 2 §4.5). Table 14 compares the best single checkpoint to a three-way soup on six mid-training mixes at 7B: the soup is equal or better on OLMES (MCF), OLMES-Gen and MMLU (MCF) in all six, for example mix A 75.6 → 77.0 OLMES and mix B 61.5 → 62.7 MMLU. GSM* is the exception: on mix E it falls from 60.5 to 43.0 and on mix C it is unchanged at 66.0, so the report's claim of consistency covers the non-GSM columns.
- DeepSeek-V3 keeps an Exponential Moving Average of the parameters in CPU memory, updated asynchronously each step, and uses it for early estimation of model performance after learning-rate decay (DeepSeek-V3 §3.2.3). This is a monitoring device, not the released checkpoint, and the report gives no loss delta for it.

**What a resumable checkpoint must contain.** The following is framework-level engineering practice rather than a claim from any artifact above: model weights (sharded), optimizer state (Adam first and second moments plus master fp32 weights), learning-rate scheduler step, per-rank RNG state, dataloader iterator position, and the fp16 dynamic loss-scaler state. Omitting the optimizer state forces Adam moments to rebuild on resume; omitting the dataloader position causes tokens to be re-seen after a restart.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3 405B | 405B | pretrain-decay/anneal | annealing token budget; context; LR endpoint; checkpoint selection | final 40M tokens; 128K; linear to 0; average of annealing-phase checkpoints (Polyak) | arXiv:2407.21783 §3.4.3 | verified 2026-09-18 | no ablation reported |
| Llama 3 | all sizes | RM / SFT / preference | checkpoint selection | average of models from different data versions or hyperparameters at each stage | arXiv:2407.21783 §4.1.5 | verified 2026-09-18 | no ablation reported |
| OLMo 2 7B | 7B | mid-train | number of anneals; tokens per anneal; merge method | 3; 50B each, different random data orders; uniform average | arXiv:2501.00656 §4.5 | verified 2026-09-18 | §4.5 Table 14: three-way soup equals or beats the best single checkpoint on 6 of 6 mixes on OLMES, OLMES-Gen and MMLU |
| OLMo 2 13B / 32B | 13B, 32B | mid-train | number of anneals; tokens per anneal; merge method | 4; three at 100B plus one at 300B; uniform average | arXiv:2501.00656 §4.5 | verified 2026-09-18 | §4.5 states the 4-way average was empirically better than averaging the three 100B runs alone; the comparison itself is not tabulated |
| DeepSeek-V3 | 671B | pretrain | EMA coefficient and purpose | EMA of parameters kept in CPU memory, updated asynchronously each step, for early estimation of post-decay performance | arXiv:2412.19437 §3.2.3 | verified 2026-09-18 | no ablation reported; not the released checkpoint |
| SWA reference runs | ResNet-50/152, DenseNet-161 | fine-tune | SWA duration; LR schedule | 10 epochs, cyclical LR | arXiv:1803.05407 §4.2 | verified 2026-09-18 | §4.2 Table 2: +0.6–0.9% top-1 over the pretrained models, mean over 3 runs |

## Findings relevant to generality
- Model Soups reports that the greedy soup improves not only in-distribution ImageNet accuracy but also the average over five natural distribution shifts (90.78 → 90.94 in-distribution, 84.68 → 85.02 out-of-distribution for ViT-G; 47.83 → 50.75 out-of-distribution for CLIP ViT-B/32), so in these runs averaging did not trade robustness for fit (Tables 1 and 3).
- Averaging is not uniformly safe across metrics: OLMo 2's soup loses 17.5 points of GSM* on mix E while gaining on the other three columns (Table 14). A soup must therefore be selected on the full evaluation suite, not on an aggregate.
- Both averaging papers condition their results on a shared initialization. Neither reports averaging across different architectures or independent initializations.

## Connections
- [[lr-schedules]] — the decay or annealing phase is where the averaged checkpoints are produced.
- [[adam]] — optimizer moments are part of the checkpoint; a resume without them restarts moment estimation.
- [[dropout]] — with dropout disabled in pretraining, validation-tracked checkpoint selection is the remaining regularizer at the SFT stage.
- [[llama-3]], [[olmo-2]] — the two technical reports that document checkpoint averaging at LLM scale.
- [[small-scale-proxies-instabilities]] — loss and gradient-norm instabilities, the condition under which rollback to a clean checkpoint matters.
- [[karpathy-training-neural-net-recipe]] — general checkpointing and validation-tracking advice at the level of a single practitioner recipe.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1803.05407 (arXiv v3, 25 February 2019); https://arxiv.org/abs/2203.05482 (arXiv v3, 1 July 2022); the Llama 3 report §3.4.3 and §4.1.5; the OLMo 2 report §4.5 and Table 14; the DeepSeek-V3 report §3.2.3.
- Corrections to the previous card version:
  - "DeepSeek-V3 and MiniCPM both report measurable val-loss reduction (~0.5%) from this trick alone" and "Latent-WSD averaging (DeepSeek, MiniCPM 2024)" → the MiniCPM paper (arXiv:2404.06395) reports no checkpoint averaging or souping; it forks decay runs of different lengths from the same stable checkpoint (MiniCPM §Figure 5 discussion). DeepSeek-V3 keeps a parameter EMA for early performance estimation (§3.2.3) and reports no such number. The 0.5% figure appears in neither source and has been removed. The same error propagated to `wiki/courses/llm-training/ch-06/read.md` line 193 and should be fixed there.
  - "Beats best single model on ImageNet, often by >1% top-1" → the reported margins of the greedy soup over the best individual model are +0.16 top-1 for ViT-G (90.78 → 90.94, Table 1) and +0.65 for CLIP ViT-B/32 (80.38 → 81.03, Table 3).
  - "MiniCPM WSD figure: late-decay-phase averaging produces lower final loss than any single decay endpoint" → no such figure exists; replaced by OLMo 2 Table 14, which does make this comparison.
  - "Llama-3 paper checkpoint section: production-grade checkpoint cadence (every ~1000 steps; restart on loss spike)" and "Cadence: every 30–60 minutes of compute (Llama-3 used ~1000-step cadence at 405B)" → the Llama 3 report states only that checkpointing saves 1 MB to 4 GB of state per GPU and that the team aimed to increase checkpoint frequency (§3.3.1, Storage). It gives no cadence in steps or minutes.
  - "Llama-3 / OLMo-2 / DeepSeek-V3: all publicly document checkpoint averaging across the decay phase" → Llama 3 and OLMo 2 do; DeepSeek-V3 documents a parameter EMA used for monitoring, which is a different operation.
  - Greedy-soup pseudocode used a strict improvement test over the running soup → Recipe 1 sorts by validation accuracy and keeps a candidate if held-out validation performance improves, which is why the soup can be no worse than the best individual model on that set.
  - "Authors: classical (Prechelt 1998 early-stopping); Izmailov et al. 2018 (SWA); Wortsman et al. 2022 (Model Soups)" → replaced by a per-artifact list with full author names, years and URLs, as the card standard requires.
- Removed as unsupported by the source: "equivalent to L2 regularization of a specific implicit prior (Bishop 2006)" (not checked against Bishop 2006 in this pass); "WSD-decay averaging (LLM pretrain default in 2024–2025)" and "the modern analog used in 7B+ LLM pretraining" as general practice claims (only Llama 3 and OLMo 2 were verified); "save N checkpoints (every ~10% of decay tokens)"; "Spike recovery ... costs ~1% throughput in expectation"; "use a fixed, large (≥100k tokens for LLM) val set"; "DPO / RL fine-tuning: standard practice is to take the SFT-init's averaged weights ... as the RL initialization" (Llama 3 §4.1.5 documents averaging at each of the RM, SFT and DPO stages for one model family, which does not establish standard practice); the attribution of a specific quoted recipe to Karpathy; "patience = 3–5 evaluations, delta = 0 or 1e-4" as sourced defaults.
- Not reported by the source: Prechelt 1998 was not fetched in this pass, so no claim is attributed to it; no primary source here reports a loss or benchmark delta for Llama 3's annealing-phase averaging; OLMo 2 does not tabulate the 4-way versus 3-way comparison it describes.
