---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/olmo-core-olmo3-configs.md (planned card; not present on 2026-09-15)
source_url: https://github.com/allenai/OLMo-core/tree/66f768b223171b5fde71cf67ccefa843ab9c51cb/src/scripts/official/OLMo3
created_at: "2026-09-15"
---

# Excerpt: OLMo-core official OLMo 3 training scripts, tokenizer config, data loader, and source mixture

**Organization:** Allen Institute for AI. **Source type:** released config/code.
**Version read:** github.com/allenai/OLMo-core at commit 66f768b (2026-08-11). Line numbers refer to that commit.
**Status:** no library card existed for this slug on 2026-09-15; every quote below was read in the raw files.

## Tokenizer (src/olmo_core/data/tokenizer.py)
- L77-82: `padded_vocab_size(pad_multiple=128)` returns `pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)`; docstring: "useful to set model embeddings to this number to increase throughput."
- L85-94: `dolma2()` → `vocab_size=100278, eos_token_id=100257, pad_token_id=100277, identifier="allenai/dolma2-tokenizer"`. Padded size: 128 × 784 = 100,352 (derived).
- L30-32: `dolma2_sigdig` is documented as "The R2L dolma2 tokenizer".

## OLMo 3 7B stage-1 script (src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py)
- `DEFAULT_SEQUENCE_LENGTH = 8192`; `GLOBAL_BATCH_SIZE = 8192 * 512  # ~4M tokens`; `LR = 3e-4`.
- `TokenizerConfig.dolma2()`; `TransformerConfig.olmo3_7B(vocab_size=tokenizer_config.padded_vocab_size())  # pad to a multiple of 128`.
- Dataset: `NumpyFSLDatasetConfig.from_data_mix(DataMix.OLMo_mix_0625_official, ...)`; loader `NumpyDataLoaderConfig(global_batch_size=GLOBAL_BATCH_SIZE, seed=34521, num_workers=8)`.
- Optimizer: `SkipStepAdamWConfig(lr=LR, weight_decay=0.1, betas=(0.9, 0.95), group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))])`; `CosWithWarmup(warmup_steps=2000)`; `z_loss_multiplier=1e-5`; `max_grad_norm=1.0`.
- `max_duration=Duration.tokens(int(5e12))  # Originally scheduled for 5T`; `hard_stop=Duration.steps(int(597046))` with the comment "But at this step we decided to extend schedule to 7T. See OLMo-3-1025-7B-pretrain-2.py" (L99-101).
- Part 2 (OLMo-3-1025-7B-pretrain-2.py L104-111): `load_trainer_state=True`, `load_optim_state=True`, `max_duration=Duration.tokens(int(7e12))  # Changed from 5T -> 7T`, `hard_stop=Duration.epochs(1)`.
- Checkpoint manifest OLMo-3-1025-7B.csv: `stage1_pretraining,1000,4194304000,...` (tokens seen = steps × 4,194,304).

## OLMo 3 7B mid-training script (OLMo-3-1025-7B-midtrain.py)
`GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens`; `MAX_TOKENS = 100_000_000_000`; `LR = 0.00020712352850360292`; `SEED = 1337`; `LinearWithWarmup(warmup=0, alpha_f=0.0)`; `load_trainer_state=False`, `load_optim_state=True`.

## Mix manifests (src/olmo_core/data/mixes/*.txt)
Each line is `label,path` to a pre-tokenized `.npy` file, for example `code-meta-reasoning,preprocessed/dolma3_dolmino_0925/run1-official/{TOKENIZER}/code-meta-reasoning/part-00-00000.npy` (OLMo-midtraining-mix-0925-ingredient1-100B.txt). The official OLMo 3 scripts consume these fixed manifests.

## Data loader state (src/olmo_core/data/data_loader.py)
- L441-449 `NumpyDataLoaderBase.state_dict()` returns `dataset_fingerprint_version`, `dataset_fingerprint`, `batches_processed`, `tokens_processed`, `seed`, `epoch`.
- L451-477 `load_state_dict`: a fingerprint mismatch raises `RuntimeError("Dataset fingerprint does not match ... This will probably result in a different data order!")` unless `ignore_fingerprint_mismatch=True`; a different seed is replaced by the checkpoint's seed "for data order consistency".
- L667-673 `_build_global_indices`: `rng = get_rng(self.seed + self.epoch)`; indices are shuffled with this generator.
- L734-735 (FSL `_get_local_instance_indices`): `if self.batches_processed > 0: indices = indices[self.batches_processed :]`.
- L757-763 FSL `load_state_dict`: `self.batches_processed = self.tokens_processed // self.global_batch_size` ("Account for change in batch size / sequence length").

## Source mixture (src/olmo_core/data/source_mixture.py)
- L29-60 `SourceMixtureConfig`: `target_ratio` ("The target ratio of the source in the mixture"), `max_repetition_ratio: float = 1.0` ("can be used to upsample the source data by setting the repetition ratio > 1"), `max_source_fraction: float = 1.0`.
- L141-148: target ratios must sum to 1.0.
- L309-321: `needed_for_source = int(self.requested_tokens * source_config.target_ratio)`; `max_for_source = int((num_for_source * source_config.max_source_fraction) * source_config.max_repetition_ratio)`; if `max_for_source < needed_for_source`, raise `OLMoConfigurationError("Insufficient tokens for source ...")`.
- L343-348: `training_steps = math.ceil(self.requested_tokens / self.global_batch_size)`; `requested_instances = training_steps * num_instances_per_batch`; per-path instance counts are rounded with Hamilton's largest-remainder method (L371-400).

## How ch-11 uses it
§1 (padded vocabulary), §7 (mixture weights to token shares; resume state), Recipe rows, Common mistakes.
