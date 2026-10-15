# Watermark Robustness and Radioactivity May Be at Odds in Federated Learning

Research code for **Watermark Robustness and Radioactivity May Be at Odds in Federated Learning** by **Leixu Huang, Zedian Shao, and Teodora Baluta**, Georgia Institute of Technology.

Accepted at the **3rd Workshop on Attributing Model Behavior at Scale: Data Attribution and Provenance (ATTRIB), NeurIPS 2026**.

## Overview

Can watermarks in synthetic training data remain detectable after federated fine-tuning—and can an adversarial server remove that signal while preserving model utility?

This work studies **watermark radioactivity**: the persistence of a detectable watermark signal in a model fine-tuned on watermarked text. We compare benign federated learning (**VanillaFL**) with an active server that uses robust aggregation to filter client updates (**ActiveFL**). Our experiments show that the evaluated watermarks remain radioactive under benign aggregation but can lose detectability under an active server, revealing a trade-off between radioactivity, robustness, and utility.

The paper evaluates Kirchenbauer-style **KGW+** and Kuditipudi-style **KTH-ITS** watermarks, Pythia models, and the C4 and Alpaca datasets. The code also exposes an Aaronson-style watermark option.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `train.py` | Dataset preparation, watermark generation, federated training, evaluation, and detection |
| `config.py` | Default settings and command-line arguments |
| `utils.py` | Data loading, model loading, optimizers, generation, detection, and checkpoint utilities |
| `robust_aggregator.py` | Robust aggregation and filtering statistics for ActiveFL |
| `wm/` | Watermark generators, detectors, and Cython detection routines |
| `scripts/graph.py` | Example plots using values embedded in the script |

## Setup

```bash
git clone https://github.com/Caroline-111/FL-Watermark-Radioactivity.git
cd FL-Watermark-Radioactivity
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-compatible version of PyTorch for your system, then install the additional packages imported by the code:

```bash
python -m pip install transformers datasets accelerate numpy scipy \
  pandas nltk evaluate rouge-score Cython matplotlib
```

This is an unpinned dependency list inferred from the source imports, rather than a validated environment lockfile. A working C/C++ compiler and Python development headers are needed because the detector compiles Cython extensions through `pyximport` on import.

The training workflow requires an NVIDIA GPU with sufficient memory. It uses float32 models and generates synthetic data with `EleutherAI/pythia-2.8b-deduped`. The paper reports experiments on an 80 GB H100; this is the experimental hardware, not an established minimum requirement. Initial runs need network access to download models, datasets, NLTK resources, and the ROUGE evaluation module.

### Select your GPU

At the top of `train.py`, the code currently sets:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

Change this to the GPU index available on your machine (for example, `'0'` on a single-GPU system), or remove the assignment and set `CUDA_VISIBLE_DEVICES` externally. The assignment in the script overrides an externally supplied value.

`config.GPU` is used in cache and output directory names; it does not select the physical GPU. Update it in `config.py` if you want those names to match your selected GPU.

## Run experiments

Run all commands from the repository root. Dataset preparation and watermark generation are integrated into `train.py`.

### VanillaFL with KGW+

```bash
python train.py \
  --dataset c4 \
  --watermark kirchenbauer \
  --model_checkpoint EleutherAI/pythia-160m-deduped \
  --n_worker 30 \
  --dirty_worker 2 \
  --client_optimizer normalized \
  --server_optimizer adam
```

`--dirty_worker` is the code's name for the number of watermarking clients. With 2 of 30 clients watermarking, the client fraction is approximately 6.7%.

### ActiveFL with KGW+

Enable robust filtering with `--whether_filter`:

```bash
python train.py \
  --dataset c4 \
  --watermark kirchenbauer \
  --model_checkpoint EleutherAI/pythia-160m-deduped \
  --n_worker 30 \
  --dirty_worker 2 \
  --client_optimizer normalized \
  --server_optimizer adam \
  --whether_filter
```

### Other experimental settings

- Use `--dataset alpaca` for Alpaca.
- Use `--watermark kuditipudi` for KTH; `KTH_WATERMARK` defaults to `KTHType.ITS` in `config.py`.
- Change `--model_checkpoint` to select another compatible model.
- Use `--dirty_worker 5` or `--dirty_worker 9` to vary the fraction of watermarking clients among 30 clients.
- Use `--whether_synthetic` for the synthetic clean-client setting. The on-the-fly synthetic evaluation-data path is marked C4-only in the code.

Inspect command-line options in `Config.get_parser()` in `config.py`. Settings such as the number of rounds, learning rate, batch size, watermark bias, temperature, and random seeds are configured directly in `config.py` rather than exposed as command-line options.

## Reuse generated training data

To save the generated training dataset, add:

```bash
--whether_save_wm --dataset_path dataset_kgw_c4.json
```

To load that dataset in a subsequent run, add:

```bash
--whether_load_wm --dataset_path dataset_kgw_c4.json
```

Use matching dataset, model, watermark, client-count, and generation settings when reusing data. The saved JSON contains the processed training dataset, including clean and watermarked client data. For synthetic-data runs that load cached data, also supply the matching evaluation JSON through `--eval_dataset_path`.

## Default configuration

These are the defaults in `config.py`, not a specification of every experiment in the paper.

| Setting | Default |
| --- | --- |
| Fine-tuned model | `EleutherAI/pythia-160m-deduped` |
| Dataset | C4 (`realnewslike`) |
| Clients / watermarking clients | 30 / 2 |
| Maximum communication rounds | 500 |
| Local epochs per round | 1 |
| Client / server optimizer | Normalized SGD / Adam |
| Learning rate | `1e-5` |
| Batch size | 8 |
| Sequence length | 1024 |
| Training / total examples | 19,200 / 21,120 |
| Training seed | 42 |
| Watermark | Kirchenbauer |
| KGW gamma / delta | 0.25 / 3.0 |
| Temperature / top-p | 0.8 / 0.95 |
| N-gram setting | 4 |
| Robust filtering | Disabled |

The training loop uses validation-loss-based early stopping with patience 3. For paper experiments, consult the paper's experimental setup and Appendix H and match all relevant settings.

## Outputs

- Training and validation losses and watermark detection results are printed to the terminal. Redirect output to a log file if needed.
- Model checkpoints and the final model are saved under `config.OUTPUT_DIR`, which defaults to `output_g1_p0`.
- ActiveFL writes per-round filtering statistics to `filtering_metrics_g<GPU>_p<PROCESS>.jsonl` in the working directory.
- The `--process` argument changes cache/output naming and can help separate experiment runs. Give cached datasets and log files distinct names as well.

`scripts/graph.py` uses hard-coded numerical values; it is not a general parser for experiment logs. The repository does not currently include a standalone pipeline that recomputes every reported paper metric from saved logs.

## Reproducibility

The code sets random seeds and several determinism-related options. Reproducing a particular paper result also requires matching the model checkpoint, dataset split, watermark settings, optimizer settings, client composition, and stopping criteria. Results can vary across software versions and hardware.

Record the Git commit, installed package versions, full configuration, hardware, and generated-data files for each experiment. The installation and experiment commands above document the source workflow; they have not been validated through a full training run as part of this README preparation.

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

The watermark modules retain upstream copyright notices and references to prior implementations. Preserve these notices when redistributing the code. This snapshot does not include a root `LICENSE` file; no new license is asserted by this README.
