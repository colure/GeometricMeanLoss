# [ICIP2025] Geometric Mean Improves Loss For Few-Shot Learning

[![Python](https://img.shields.io/badge/Python-3.8.20-blue?logo=python)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1%20%2B%20cu121-orange?logo=pytorch)](https://pytorch.org/) [![arXiv](https://img.shields.io/badge/arXiv-2501.14593-b31b1b.svg)](https://arxiv.org/abs/2501.14593)

Official implementation of [**"Geometric Mean Improves Loss For Few-Shot Learning"**](https://arxiv.org/abs/2501.14593), by Tong Wu and Takumi Kobayashi.

## About

Few-shot learning (FSL) is a challenging task in machine learning, demanding a model to render discriminative classification by using only a few labeled samples. In the literature of FSL, deep models are trained in a manner of metric learning to provide metric in a feature space which is well generalizable to classify samples of novel classes; in the space, even a few amount of labeled training examples can construct an effective classifier. In this paper, we propose a novel FSL loss based on geometric mean to embed discriminative metric into deep features. In contrast to the other losses such as utilizing arithmetic mean in softmax-based formulation, the proposed method leverages geometric mean to aggregate pair-wise relationships among samples for enhancing discriminative metric across class categories. The proposed loss is not only formulated in a simple form but also is thoroughly analyzed in theoretical ways to reveal its favorable characteristics which are favorable for learning feature metric in FSL. In the experiments on few-shot image classification tasks, the method produces competitive performance in comparison to the other losses.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
git clone https://github.com/koko9tsu/GeometricMeanLoss.git
cd GeometricMeanLoss

# Initialize and install dependencies automatically
uv sync
```

## Usage

### 1. Data Preparation

#### Datasets

- **miniImageNet**: [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)
- **CIFAR-FS**: [Meta-learning with differentiable closed-form solvers](https://arxiv.org/abs/1805.08136)
- **tieredImageNet**: [Meta-Learning for Semi-Supervised Few-Shot Classification](https://arxiv.org/abs/1803.00676)
- **CUB-200-2011**: [Caltech-UCSD Birds dataset](https://www.vision.caltech.edu/datasets/)


Place datasets in the `./data` directory (e.g., `mini_imagenet`, `cifar_fs`). You can adjust default paths in `config.toml`.

### 2. Configuration
- **`config.toml`**: Centralized configuration for all hyperparameters and paths.
- **CLI Overrides**: Override any TOML setting via command line (e.g., `--batch-size 64`).

### 3. Run
We utilize [Distributed Data Parallel (DDP)](https://docs.pytorch.org/docs/stable/elastic/run.html) for multi-GPU training. Use `uv run` to execute with the managed environment:
```bash
# To train the model:
uv run torchrun --nproc_per_node=${num_gpus} train.py --loss GMLoss --logit l1_dist --output-dir ./result/my_experiment

# To evaluate a checkpoint:
uv run python train.py --resume ./result/my_experiment/checkpoints/best_shot5_model.pth --test-only
```
More examples can be found in `train_script.sh`.

## Results

<table style="border-collapse: collapse; width: 100%; text-align: center; font-family: 'Times New Roman', serif; color: black;">
  <thead style="border-top: 2px solid black;">
    <tr>
      <th rowspan="2" style="vertical-align: middle; padding: 6px; border-bottom: 1.5px solid black;">Loss</th>
      <th colspan="2" style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">miniImageNet</th>
      <th colspan="2" style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">CIFAR-FS</th>
      <th colspan="2" style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">tieredImageNet</th>
    </tr>
    <tr>
      <th style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">1-shot</th>
      <th style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">5-shot</th>
      <th style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">1-shot</th>
      <th style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">5-shot</th>
      <th style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">1-shot</th>
      <th style="padding: 6px; border-bottom: 1.5px solid black; text-align: center;">5-shot</th>
    </tr>
  </thead>
  
  <tbody style="border-bottom: 2px solid black;">
    <tr>
      <td style="padding: 6px;">PN</td>
      <td style="padding: 6px;">62.42±0.20</td>
      <td style="padding: 6px;">79.13±0.15</td>
      <td style="padding: 6px;">67.14±0.22</td>
      <td style="padding: 6px;">82.36±0.16</td>
      <td style="padding: 6px;">66.74±0.23</td>
      <td style="padding: 6px;">82.14±0.17</td>
    </tr>
    <tr>
      <td style="padding: 6px;">NCA</td>
      <td style="padding: 6px;">62.68±0.20</td>
      <td style="padding: 6px;">78.93±0.54</td>
      <td style="padding: 6px;">69.20±0.21</td>
      <td style="padding: 6px;">84.24±0.16</td>
      <td style="padding: 6px;">67.21±0.22</td>
      <td style="padding: 6px;">83.77±0.16</td>
    </tr>
    <tr>
      <td style="padding: 6px; border-top: 1.5px solid black;">GML</td>
      <td style="padding: 6px; border-top: 1.5px solid black;"><strong>65.51±0.20</strong></td>
      <td style="padding: 6px; border-top: 1.5px solid black;"><strong>81.13±0.14</strong></td>
      <td style="padding: 6px; border-top: 1.5px solid black;"><strong>71.09±0.22</strong></td>
      <td style="padding: 6px; border-top: 1.5px solid black;"><strong>85.08±0.16</strong></td>
      <td style="padding: 6px; border-top: 1.5px solid black;"><strong>69.61±0.23</strong></td>
      <td style="padding: 6px; border-top: 1.5px solid black;"><strong>84.04±0.16</strong></td>
    </tr>
  </tbody>
</table>

## Citation

If you find this project or paper useful in your research, please consider citing the original paper:
```bibtex
@misc{wu2025geometricmeanimprovesloss,
      title={Geometric Mean Improves Loss For Few-Shot Learning},
      author={Tong Wu and Takumi Kobayashi},
      year={2025},
      eprint={2501.14593},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.14593},
}
```
## Acknowledgments

Parts of this code build upon the following projects:
- [On-episodes-fsl](https://github.com/fiveai/on-episodes-fsl)
- [vision](https://github.com/pytorch/vision)

## License
MIT License.