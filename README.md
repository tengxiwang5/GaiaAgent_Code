# 🎉 [ISPRS JPRS 2026] Towards Comprehensive Multi-task Land Cover Change Detection Leveraging Vision-Language Model and LLM-driven Agents

Official implementation of:

**Towards comprehensive multi-task land cover change detection leveraging vision-language model and LLM-driven agents**

Tengxi Wang, Yiru Wang, Shuai Zhang, Yuxuan Liang, and Wufan Zhao

ISPRS Journal of Photogrammetry and Remote Sensing, Volume 238, 2026, Pages 756-774

[[Paper]](https://doi.org/10.1016/j.isprsjprs.2026.05.025) [[ScienceDirect]](https://www.sciencedirect.com/science/article/pii/S0924271626002662)

---

## 🔥 NEWS

- [2026.07] We release the code of GaiaAgent for multi-task land cover change detection.
- [2026.05] The paper was published in ISPRS Journal of Photogrammetry and Remote Sensing.

---

## ✨ Overview

We present GaiaAgent, a vision-language framework for comprehensive multi-task land cover change detection. The released implementation includes the following core modules:

- `GaiaAgent_VLMTool`: the main model for multi-task land cover change detection.
- `InformationInteraction`: cross-temporal feature interaction between bi-temporal observations.
- `SemanticPositioning`: semantic positioning with adaptive positional encoding and Transformer encoding.
- `ContrastiveLearning`: text-image contrastive learning used during training.

Text descriptions are used only during training. Validation and testing use image pairs only.

## 📁 Code Structure

```text
GaiaAgent_Code/
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
├── augmentations.py
└── utils/
```

## 🗂️ Dataset Structure

Please organize the dataset as follows:

```text
data/
├── train/
│   ├── p1/       # pre-change images
│   ├── p2/       # post-change images
│   ├── 2d/       # 2D land-cover change labels
│   ├── 3d/       # 3D regression labels
│   └── text/     # text descriptions, training only
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

Notes:

- File names should be aligned across `p1/`, `p2/`, `2d/`, and `3d/`.
- The `text/` folder is required only for the training split.
- Validation and test dataloaders do not load text.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

The MobileSAM checkpoint is not included in this repository. Please place `mobile_sam.pt` in the project root before running:

```text
GaiaAgent_Code/
└── mobile_sam.pt
```

## ⚙️ Configuration

Edit [config/config.yaml](config/config.yaml) before running.

Example:

```yaml
data:
  train:
    path: '/path/to/train/'
    text_path: '/path/to/train/'
    batch_size: 2
  val:
    path: '/path/to/val/'
  test:
    path: '/path/to/test/'
```

The 2D focal loss is controlled by:

```yaml
model:
  2d_loss: 'focal'
  2d_loss_weights: [0.3, 2.0, 2.0, 3.0, 1.5, 0.8, 1.0]
  focal_alpha: 0.5
  focal_gamma: 2
```

## 🚀 Training

```bash
python train.py -c config
```

Checkpoints and logs are saved under:

```text
results/config/
```

## 🧪 Testing

```bash
python test.py -c config
```

## 📦 Checkpoints

Large files such as `*.pth`, `*.pt`, and `mobile_sam.pt` are not tracked in git. Model weights will be provided separately if released.

## 📬 Contact

If you have questions, please open an issue in this repository.

## 📝 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{WANG2026756,
  title = {Towards comprehensive multi-task land cover change detection leveraging vision-language model and LLM-driven agents},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {238},
  pages = {756-774},
  year = {2026},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2026.05.025},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271626002662},
  author = {Tengxi Wang and Yiru Wang and Shuai Zhang and Yuxuan Liang and Wufan Zhao}
}
```
