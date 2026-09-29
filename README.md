# Dynamic Forward-Pass Scheduling for Instance-Dependent Early Stopping

Research code and results from my 2025 project at the University of Sydney AI Centre. I studied whether samples already identified as *mastered* by Instance-Dependent Early Stopping (IES) need to be re-evaluated at every epoch. The project introduces dynamic schedules that reduce those redundant forward passes while controlling the accuracy–compute trade-off.

> **My contribution.** I designed and benchmarked static, epoch-based, learning-rate-aware, and feedback-based forward-pass schedules; evaluated a late-training switch from second- to first-order loss differences; and investigated the feasibility of applying IES to supervised fine-tuning of causal language models.

This work builds on the ICLR 2025 Spotlight paper [Instance-dependent Early Stopping](https://arxiv.org/abs/2502.07547) by Suqin Yuan, Runqi Lin, Lei Feng, Bo Han, and Tongliang Liu. The original authors' [reference implementation](https://github.com/tmllab/2025_ICLR_IES) should be cited alongside this project.

## Main result

The learning-rate-aware schedule was the most robust strategy tested. It increases the re-evaluation interval as optimizer updates become smaller, without introducing an additional tuning parameter. Across the four principal settings below, it retained accuracy within the reported experimental uncertainty and reduced measured wall-clock training time.

| Dataset | LR decay | Constant schedule | LR-aware schedule | Time reduction | Accuracy (constant → LR-aware) |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | Linear | 3,836.2 s | 3,268.1 s | 14.8% | 95.1% → 95.1% |
| CIFAR-10 | Exponential | 2,937.0 s | 2,136.9 s | 27.2% | 94.7% → 94.7% |
| CIFAR-100 | Linear | 4,490.5 s | 4,047.1 s | 9.9% | 77.3% → 77.3% |
| CIFAR-100 | Exponential | 4,061.3 s | 3,859.9 s | 5.0% | 76.9% → 76.6% |

Values are means over five runs; full confidence intervals and alternative schedules are in [`results/benchmark_results.csv`](results/benchmark_results.csv) and the [project report](Internship_report_Felix_Azian_2025.pdf). Wall-clock results are hardware- and implementation-dependent.

## Method

IES tracks each training instance's loss trajectory and treats a sample as mastered when a smoothed second-order finite difference remains near zero. Mastered samples are removed from backpropagation but must periodically be checked in case the model forgets them.

This project changes the re-check period, \(k_t\):

- **Static:** fixed periods from 1 to 10.
- **Epoch-based:** linear, logarithmic, and square-root growth.
- **Adaptive ratio:** feedback from the change in the mastered-sample ratio over ten-epoch windows.
- **Learning-rate-aware:** a piecewise schedule that raises \(k_t\) as the learning rate decays.

The final experiments also switch from the second-order to the first-order loss difference once the schedule reaches its maximum period. That switch saved more computation in some settings but was less stable and generally reduced accuracy, so it was not selected as the primary method.

## Repository structure

```text
.
├── src/dynamic_ies/          # Tested scheduling and mastery-criterion components
├── scripts/train_cifar.py    # Reproducible CIFAR-10/100 experiment entry point
├── tests/                    # Unit tests for the research logic
├── results/                  # Results reported in the internship report
└── Internship_report_...pdf  # Non-confidential research report
```

Raw CIFAR files, model checkpoints, generated logs, and complete copies of LLaMA-Factory/DeepSpeed are intentionally excluded. They are data, generated artifacts, or third-party dependencies—not authored source code.

## Reproduce an experiment

Python 3.10+ and a CUDA-capable GPU are recommended. The script downloads CIFAR through `torchvision` on first use.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .

python3 scripts/train_cifar.py \
  --dataset cifar10 \
  --schedule lr-aware \
  --lr-decay exponential \
  --epochs 200 \
  --runs 5
```

Useful comparisons:

```bash
# Original IES-style re-evaluation at every epoch
python3 scripts/train_cifar.py --dataset cifar10 --schedule constant --period 1

# Square-root epoch schedule
python3 scripts/train_cifar.py --dataset cifar10 --schedule sqrt

# Disable sample exclusion (ordinary training baseline)
python3 scripts/train_cifar.py --dataset cifar10 --threshold 0
```

Run the lightweight tests with:

```bash
python3 -m unittest discover -s tests -v
```

Exact wall-clock values can differ from the report because this repository is a cleaned reimplementation using current `torchvision` dataset/model APIs. Seeds, aggregate statistics, and run metadata are written to `outputs/`.

## LLM fine-tuning study

I also ran full supervised fine-tuning of Llama 3.2 1B on BioInstruct using LLaMA-Factory and four RTX 4090 GPUs. This phase identified two obstacles rather than claiming a finished method: typical SFT runs provide too few epochs for stable second-order trajectories, and variable sequence lengths make per-example loss signals heteroscedastic. The report discusses possible normalization and smoothing directions. Model weights, checkpoints, and third-party framework sources are not redistributed here.

## Attribution and reuse

This repository is an academic portfolio artifact and currently has **no open-source license**. The upstream IES reference repository also does not publish an explicit license at the time of this cleanup, so no license is inferred. Please contact the respective authors before reusing code. See [`NOTICE.md`](NOTICE.md) for provenance and citations.
