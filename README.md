# GaiaAgent

Official implementation of the ISPRS Journal of Photogrammetry and Remote Sensing paper:

**GaiaAgent: A Vision-Language Agent for Change Detection**

> Paper link, DOI, and citation will be updated after publication.

## Overview

GaiaAgent is a vision-language change detection framework. The current implementation contains:

- `GaiaAgent_VLMTool`: the main change detection model.
- `InformationInteraction`: cross-temporal feature interaction.
- `SemanticPositioning`: adaptive semantic positioning with learnable positional encoding and Transformer encoding.
- `ContrastiveLearning`: text-image contrastive learning used during training.

Text descriptions are used only during training. Validation and testing use image pairs only.

## Repository

```text
.
├── models/
│   ├── GaiaAgent_VLMTool.py
│   ├── InformationInteraction.py
│   ├── SemanticPositioning.py
│   └── ContrastiveLearning.py
├── config/
│   ├── config.yaml
│   └── config.example.yaml
├── train.py
├── test.py
├── dataloader.py
├── losses.py
├── optim.py
└── utils/
```

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

This repository does not include the MobileSAM checkpoint. Please place `mobile_sam.pt` in the project root before running the code.

## Data

The expected dataset structure is:

```text
data/
├── train/
│   ├── p1/
│   ├── p2/
│   ├── 2d/
│   ├── 3d/
│   └── text/
├── val/
│   ├── p1/
│   ├── p2/
│   ├── 2d/
│   └── 3d/
└── test/
    ├── p1/
    ├── p2/
    ├── 2d/
    └── 3d/
```

Only the training split requires the `text/` folder.

## Configuration

Edit [config/config.yaml](config/config.yaml) before running:

```yaml
data:
  train:
    path: '/path/to/train/'
    text_path: '/path/to/train/'
  val:
    path: '/path/to/val/'
  test:
    path: '/path/to/test/'
```

The 2D focal loss is controlled from the config:

```yaml
model:
  2d_loss: 'focal'
  2d_loss_weights: [0.3, 2.0, 2.0, 3.0, 1.5, 0.8, 1.0]
  focal_alpha: 0.5
  focal_gamma: 2
```

## Training

```bash
python train.py -c config
```

Checkpoints and logs are saved under:

```text
results/config/
```

## Testing

```bash
python test.py -c config
```

## Checkpoints

Large files such as `*.pth` and `mobile_sam.pt` are not tracked in git. If checkpoints are released, they will be provided separately.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{wang2026gaiaagent,
  title   = {GaiaAgent: A Vision-Language Agent for Change Detection},
  author  = {TBD},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  doi     = {TBD}
}
```

## License

The license will be updated before the final public release.
