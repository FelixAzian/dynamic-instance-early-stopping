# Provenance and attribution

## Project contribution

Felix Azian conducted this research project at the University of Sydney AI Centre from May to August 2025 under the supervision of Suqin Yuan, with academic supervision from Zacharie Ales at ENSTA Paris.

The project's original contributions are the design, implementation, and empirical comparison of dynamic forward-pass schedules for mastered samples; the derivative-order switching study; and the feasibility analysis for supervised fine-tuning of causal language models.

## Upstream research

This project extends:

> Suqin Yuan, Runqi Lin, Lei Feng, Bo Han, and Tongliang Liu. “Instance-dependent Early Stopping.” *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025. https://arxiv.org/abs/2502.07547

Reference implementation: https://github.com/tmllab/2025_ICLR_IES

## Resulting publication

This research contributed to:

> Suqin Yuan, Runqi Lin, Felix Azian, Lei Feng, Bo Han, Gang Niu, Masashi Sugiyama, and Tongliang Liu. “Instance-Dependent Early Stopping for Adaptive Data Pruning.” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2026. https://doi.org/10.1109/TPAMI.2026.3693108

IEEE Computer Society publication page: https://www.computer.org/csdl/journal/tp/2026/10/11517544/2gswlhfa9tm

The clean implementation in this repository calls `torchvision`'s ResNet API and does not redistribute the upstream repository's unchanged `resnet.py`. Because the upstream repository does not state a software license, this repository does not infer one.

## Third-party tools used in the exploratory study

- LLaMA-Factory: https://github.com/hiyouga/LlamaFactory (Apache-2.0)
- DeepSpeed: https://github.com/microsoft/DeepSpeed (Apache-2.0)
- CIFAR-10 and CIFAR-100: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.

No model weights, dataset copies, generated checkpoints, or vendored framework sources are included in the Git repository.
