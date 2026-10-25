# Watermark Robustness and Radioactivity May Be at Odds in Federated Learning

Research code for **Watermark Robustness and Radioactivity May Be at Odds in Federated Learning** by **Leixu Huang, Zedian Shao, and Teodora Baluta**, Georgia Institute of Technology.

Accepted at the **3rd Workshop on Attributing Model Behavior at Scale: Data Attribution and Provenance (ATTRIB), NeurIPS 2026**.

## Overview

Can watermarks in training data remain detectable after federated fine-tuning? Can an adversarial server remove watermark while preserving model utility?

This work studies **watermark radioactivity** in federated learning (FL). We compare benign federated learning (**VanillaFL**) with an active server that uses strong robust aggregation to filter client updates (**ActiveFL**). Our experiments show that the evaluated watermarks remain radioactive under VanillaFL but lose detectability under ActiveFL, revealing a trade-off between watermark radioactivity, robustness, and utility.

The paper evaluates Kirchenbauer-style **KGW+** and Kuditipudi-style **KTH-ITS** watermarks. The code also exposes an Aaronson-style watermark option.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `train.py` | Main file |
| `config.py` | Default configurations and command-line arguments |
| `utils.py` | Helper functions |
| `robust_aggregator.py` | Robust aggregation (RandEigen) |
| `wm/` | Watermark generators and detectors |
| `scripts` | Scripts for reproducing the experiments reported in the paper’s tables and figures |

## Setup

```bash
git clone https://github.com/Caroline-111/FL-Watermark-Radioactivity.git
cd FL-Watermark-Radioactivity
conda env create -f environment.yml
conda activate watermark
```

The paper reports experiments on an 80 GB H100; this is the experimental hardware used in the study, not a strict minimum requirement. Results may vary slightly with different GPU hardware or package versions.

## Run experiments

The repo includes experiment launch scripts in [scripts/experiments](scripts/experiments). These are Slurm-based examples used to run the paper’s training jobs. 
A typical direct launch looks like:

```bash
python train.py \
  --process 0 \
  --dataset_path "./dataset_c4_KGW.json" \
  --dataset c4 \
  --model_checkpoint "EleutherAI/pythia-70m-deduped" \
  --watermark kirchenbauer
```

The exact experiment command depends on the dataset, model, and watermark type; the scripts in [scripts/experiments](scripts/experiments) are the reference workflow.

## Reuse generated training data

To save the generated training dataset, add:

```bash
--whether_save_wm --dataset_path dataset.json
```

To load that dataset in a subsequent run, add:

```bash
--whether_load_wm --dataset_path dataset.json
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{huang2026watermark,
  title={Watermark Robustness and Radioactivity May Be at Odds in Federated Learning},
  author={Huang, Leixu and Shao, Zedian and Baluta, Teodora},
  booktitle={3rd Workshop on Attributing Model Behavior at Scale: Data Attribution and Provenance, NeurIPS},
  year={2026}
}
```

## Attribution and licensing

This repository incorporates adapted code from upstream projects, including:

- the watermark implementation from https://github.com/facebookresearch/radioactive-watermark
- the robust aggregation implementation from https://github.com/dezhanglee/randeigen_artifacts

The watermark modules and aggregation-related code retain upstream copyright notices and references to prior implementations. Preserve those notices when redistributing this code. This snapshot does not include a root `LICENSE` file, and no new license is asserted by this README; please consult the upstream repositories for their original licensing terms.
