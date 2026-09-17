<!-- scope: chapter-local excerpt for ch-22; no library card existed at the 2026-09 revision
     source: Zhao, Andriushchenko, Croce, Flammarion. "Long Is More for Alignment: A Simple but Tough-to-Beat Baseline for Instruction Fine-Tuning". arXiv:2402.04833 (v1 2024-02; ICML 2024)
     checked: 2026-09-15 against https://arxiv.org/abs/2402.04833 (v2, 4 Jun 2024)
-->

# Excerpt: Long Is More for Alignment

- **Core result.** Fine-tuning on the 1,000 examples with the longest responses from Alpaca-52k or Evol-Instruct-70k beats AlpaGasus-1k and LIMA-1k in GPT-4 and PaLM-2 head-to-head judging, and stays competitive on Open LLM benchmarks (Abstract, Fig. 1).
- **Source type:** paper. **Reliability:** single study; judge-based evaluation plus Open LLM tasks and 425 human preferences (§4.1).

## Head-to-head (Llama-2-7B, Fig. 1a; wins / ties / losses for Alpaca-1k-longest, %)
| Opponent | GPT-4 judge | PaLM-2 judge |
|---|---|---|
| Alpaca-52k | 72.4 / 12.4 / 15.2 | 78.1 / 15.2 / 6.7 |
| AlpaGasus-1k | 67.7 / 16.5 / 15.8 | 71.9 / 20.8 / 7.3 |
| LIMA-1k | 46.3 / 25.4 / 28.3 | 47.0 / 26.9 / 26.1 |

Mean training-response length (tokens): Alpaca-52k 51.4, Alpaca-1k-longest 256.8, LIMA-1k 531.3 (Fig. 1b, Fig. 6c). Human preferences: Alpaca-1k-longest 71.0% win rate over 425 judgments (§4.1).

## AlpacaEval 2.0 (Table 1, win rate %, average length)
Llama-2-7B: AlpaGasus-1k 2.69 (745); LIMA-1k 2.74 (1360); Alpaca-52k 2.74 (586); Alpaca-1k-longest 3.16 (1810); Refined-Alpaca-1k-longest 6.00 (1732); + NEFTune 7.88 (1801); + NEFTune + max generation 4096 7.83 (2478).
Mistral-7B-v0.1: Alpaca-52k 3.42; AlpaGasus-1k 4.91; LIMA-1k 6.76; Alpaca-1k-longest 7.13; Refined 11.74.

## MT-Bench (Table 2), Llama-2-7B
Alpaca-52k 3.74; AlpaGasus-1k 3.63; LIMA-1k 3.95; Alpaca-1k-longest 3.96; Refined 4.18.

## Length controls (§5, Fig. 6–7)
- Postponing EOS until token 150 makes Alpaca-52k and AlpaGasus-1k outputs similar in length; GPT-4 and PaLM-2 still prefer Alpaca-1k-longest (Fig. 6a).
- Prompting Mistral-7B models to equalize length (Alpaca-52k-prompt 197 tokens vs Alpaca-1k-longest 184) still favours Alpaca-1k-longest (Fig. 6b).
- During fine-tuning, response length decreases after early epochs while win rate rises (Fig. 7).
- The authors conclude the results are not due to judge length bias (§6) (author interpretation).

## Open LLM (Fig. 5, Llama-2-7B, average of ARC, MMLU, TruthfulQA, Winogrande)
Base 53.1; LIMA-1k 55.9; Refined-Alpaca-1k-longest 56.4 (56.5 with NEFTune); AlpaGasus-1k and Alpaca-1k-longest about 1 point above base. HellaSwag excluded because LIMA-1k contains its examples; GSM8K excluded (§4.3).

## Training settings (App. A, Table 3)
1k-sample runs on Llama-2-7B: 4 GPUs, 15 epochs, LR 1e-5, linear schedule, batch 128, context 2048, weight decay 0.1, warmup 0.0. Mistral-7B-v0.1 1k runs: LR 2e-6. NEFTune noise 5 (Llama-2-7B), 3 (Mistral-7B, Llama-2-13B).
