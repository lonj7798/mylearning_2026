<!-- excerpt for [[ch-59]] — SWE-smith rejection-sampling SFT setup and the Python-only generality result
     source: SWE-smith: Scaling Data for Software Engineering Agents, arXiv:2504.21798v2 (§3, §4, App. F.1, F.4)
     read 2026-09-17 from the cached primary text. No library card exists for this artifact yet.
-->

# SWE-smith — rejection-sampling SFT setup and results

## Data and procedure (§2, §3)

- "Using SWE-smith, we create a dataset of 50k instances sourced from 128 GitHub repositories" (Abstract).
- Expert model: `claude-3-7-sonnet-20250219` run inside SWE-agent, "at most 75 steps and $2.00 cost limit"
  per instance; student inference uses the same 75-step maximum at temperature 0.0 (§3).
- Student bases: Qwen-2.5-Coder-Instruct 7B and 32B (§3).
- Expert trajectories were attempted for 8,686 unique task instances, "or 17.3% of the SWE-smith dataset";
  "The final pool of 6,457 represents a 36% resolve rate of all 17,906 attempts" (§4).
- Filtering: "we limit the number of times any SWE-smith task instance is represented in the training set to
  3 trajectories. This leads to the final 5,016 training set." The reason given is that "'easier'
  trajectories — task instances that are repeatedly solved across multiple runs — degrade model
  performance" (§4).

## Training hyper-parameters (App. F.1)

> "We perform full parameter fine tuning using the torchtune (PyTorch, 2024) library, with learning rate
> 5e-5, maximum 3 epochs, and max context length of 32768. Training was carried on Modal (Modal, 2025) on
> 2-8 NVIDIA H100 80G GPUs."

Only trajectories corresponding to resolved instances are trained on (§3, App. F.1).

## Table 3 (§4): pass@1 resolve rate, no verifiers and no multiple attempts

| Model | System | Train size | SWE-bench Lite | SWE-bench Verified |
|---|---|---|---|---|
| Claude 3.7 Sonnet | SWE-agent | – | 48.0 | 58.2 |
| SWE-gym-32B | OpenHands | 491 | 15.3 | 20.6 |
| R2E-Gym-32B | OpenHands | 3.3k | – | 34.4 |
| SWE-agent-LM-7B | SWE-agent | 2k | 11.7 | 15.2 |
| SWE-agent-LM-32B | SWE-agent | 5k | 30.7 | 40.2 |

Matched-size comparison (§4): fine-tuning the 32B model on 500 successful trajectories gives 28.2% on
SWE-bench Verified, "a relative difference of +8.2% with Pan et al. (2024) and +0.7% with Jain et al. (2025)".

## App. F.4: the out-of-distribution result

> "Out of 300 task instances, we found that Claude 3.7 Sonnet achieved a 43% Pass@1 resolve rate, which is
> significantly better than SWE-agent-LM-32B (8.4%) and Qwen 2.5 Coder Instruct (6.5%). SWE-agent-LM-32B does
> not demonstrate a significant improvement over the baseline model."

SWE-bench Multilingual consists of 300 task instances covering 9 programming languages other than Python
(§3, App. F.2). The authors' stated working hypothesis is that rejection-sampling fine-tuning "had improved
its ability to carry out multi-turn interactions in this task setting", while "there were instances where
code edits reflected syntax closer to Python despite code and files viewed in previous steps clearly not
being written in Python" (App. F.4).

Figure 22: SWE-agent-LM-32B pass@k on SWE-bench Verified rises from 40.2 at k = 1 to 54.8 at k = 6.
Figure 23: rejection-sampling selection against unfiltered random sampling of trajectories at
n = 100/200/400/800/1600 training points gives 14.3/22.4/27.8/30.1/33.4 against 10.2/19.7/18.3/23.4/27.8
percent resolved.
