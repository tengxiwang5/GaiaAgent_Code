# GaiaAgent_Code
The code for Towards Comprehensive Multi-Task Land Cover Change Detection Leveraging Vision-Language Model and LLM-Driven Agents

The source code will be made publicly available soon.


# GaiaAgent VLMTool

GaiaAgent VLMTool is a change detection model with visual-language contrastive learning, semantic positioning, and information interaction modules.

## Structure

- `models/GaiaAgent_VLMTool.py`: main model.
- `models/InformationInteraction.py`: two-temporal feature interaction module.
- `models/ContrastiveLearning.py`: text-image contrastive learning module.
- `models/SemanticPositioning.py`: semantic positioning module.
- `train.py`: training entry.
- `test.py`: evaluation entry.
- `dataloader.py`: dataset loader. Text is loaded only in training mode.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the config:

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` to point to your dataset paths.

The MobileSAM checkpoint is not included in this repository. Place `mobile_sam.pt` in the project root before running.

## Run

Train:

```bash
python train.py -c config
```

Test:

```bash
python test.py -c config
```
