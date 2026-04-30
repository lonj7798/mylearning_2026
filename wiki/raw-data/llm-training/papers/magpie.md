<!-- scope: extract alignment data from aligned LLMs via pre-query-only generation, filtering, and multi-turn / DPO extensions
     deps: [[self-instruct]]
     see-also: [[persona-hub]], [[evol-instruct]], [[model-collapse]], [[prismatic-synthesis]]
-->

# Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing
- **Core Insight:** An aligned chat model can leak plausible user instructions when you prompt only the pre-query template, so instruction data can be synthesized directly from an open-weight aligned model without seed prompts or API calls.
- **Guideline:** For SFT data bootstrapping, sample an aligned teacher on the pre-query template alone to generate instructions, wrap those instructions back into the chat template to generate responses, then filter by quality, difficulty, diversity, reward, and safety.
- **Authors:** Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin
- **Year:** 2025
- **URL:** https://proceedings.iclr.cc/paper_files/paper/2025/hash/be06e3802e9411381feece79b4d960c1-Abstract-Conference.html
- **Relevant topics:** synthetic alignment data, zero-seed synthesis, filtering, diversity, scale, instruction tuning

## Abstract
Magpie asks whether high-quality alignment data can be synthesized from aligned LLMs themselves. The key trick is to feed the model only the left side of the chat template, so the model auto-regressively generates a user instruction on its own. Using this "nothing" prompt on Llama-3-Instruct yields millions of instructions and matching responses, plus extensions for multi-turn, preference-optimization, domain-specific, and multilingual data. The paper argues that Magpie SFT data can outperform older public instruction sets, and can even approach or beat official instruction-tuned models on alignment benchmarks despite using far fewer downstream training examples.

## Key Contributions
- Introduced a zero-seed, zero-human pipeline for extracting instructions from aligned LLMs using only the pre-query template.
- Generated 4M instruction-response pairs and then curated 300K filtered subsets for Air/Pro variants.
- Showed that Magpie SFT alone can beat public SFT + preference-optimization baselines on AlpacaEval 2, Arena-Hard, and WildBench.
- Released extensions for multi-turn data, DPO-style preference data, domain-specific data, and multilingual data.
- Quantified the tradeoff between raw scale and curated quality through explicit filter recipes and cost accounting.

## Key Figures/Tables to Study
- **Figure 1** - the two-step pipeline: pre-query-only instruction generation, then response generation using the completed chat template.
- **Figure 3** - minimum neighbor distance and reward-difference analysis, which explain why the filtered sets are less repetitive and more useful.
- **Table 4** - dataset-scale comparison against Alpaca, Evol-Instruct, UltraChat, ShareGPT, and other alignment corpora.
- **Table 5** - the off-the-shelf filter configurations for selecting 300K subsets.

## Technical Details
- **Step 1:** use the aligned model's chat template pre-query prefix only. For Llama-3, the paper gives `Tpre-query = <|start_header_id|>user<|end_header_id|>` and stops generation at EOS.
- **Step 2:** feed the generated instruction back through the full user/assistant template to synthesize the response.
- **Scale:** MAGPIE-Air is 3M raw instruction-response pairs from Llama-3-8B-Instruct; MAGPIE-Pro is 1M from Llama-3-70B-Instruct. The paper reports 206 GPU hours for Air and 614 GPU hours for Pro.
- **Filtering metrics:** input length, output length, task category, input quality, input difficulty, minimum neighbor distance, reward, and reward difference.
- **Scoring setup:** quality is rated on a 1-5 scale from "very poor" to "excellent"; difficulty on a 1-5 scale from "very easy" to "very hard"; minimum neighbor distance uses `all-mpnet-base-v2` embeddings plus FAISS; response quality uses `FsfairX-LLaMA3-RM-v0.1`; safety uses `Llama-Guard-2`.
- **Thresholds:** the paper sets `tau1 = -12` and `tau2 = 0`; output-length filtering is applied last and keeps the longest responses.
- **Representative filters:** MAGPIE-Air-300K keeps longest outputs with input quality >= good, difficulty >= medium, positive min-neighbor distance, and reward-difference > tau2; MAGPIE-Pro variants relax or swap these constraints to produce 300K, 338K, or 200K curated slices.
- **Safety result:** less than 1% of MAGPIE-Air and MAGPIE-Pro is flagged as potentially harmful.

## Connections
- Contrasts with [[self-instruct]] and [[evol-instruct]] by removing seed questions and prompt-engineering overhead.
- Complements UltraChat-style dialogue synthesis and [[persona-hub]]: Magpie maximizes extraction efficiency and can be combined with other diversity schemes.
- Useful for [[model-collapse]] discussions because the data comes from a narrower teacher manifold than human-written instruction corpora.
- The filtered subsets are an ingredient for later open post-training recipes and small aligned models built from synthetic alignment data.
