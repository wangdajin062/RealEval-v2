# H100_package_realeval

A real-computation evaluation suite (RealEval) for fraud detection scenarios, running on H100 GPUs. Covers knowledge distillation, privacy evaluation, speculative decoding, and more.

## Project Structure

```
├── realeval/          # Core library (data loading, model loading, distillation, metrics, privacy, etc.)
├── experiments/       # 14 experiments + unified runner
├── config/            # Configuration files (experiment parameters, paper reference values, RunPod config)
├── cluster/           # Cluster deployment scripts (SLURM, RunPod, DDP)
├── tests/             # Unit tests
├── data/scripts/      # Build & data-prep scripts
├── outputs/           # Output results (result JSONs, figures, tables)
└── data/              # Datasets organized by source
    ├── TAF28k/        # TeleAntiFraud-28k (SFT, binary classification, audio)
    │   ├── sft/
    │   ├── binary_classification/
    │   └── audio/
    ├── ChiFraud/      # ChiFraud (text classification, audio)
    │   ├── dataset/
    │   └── audio/
    └── AdvFraud3k/    # AdvFraud-3k adversarial dataset
        ├── advfraud3k.json
        └── advfraud3k.jsonl
```

## Quick Start

```bash
# Install dependencies
bash setup.sh

# Smoke test (verify code runs)
bash run.sh --smoke

# Hardware check
python -m experiments.runner --check

# Run specific experiments
python -m experiments.runner --exp 1,3,6
```

## Experiment List

| Exp   | Description                                                                   |
| ----- | ----------------------------------------------------------------------------- |
| exp1  | QAD Production Distillation (teacher→student KL + OV-Freeze)                 |
| exp2  | Loss Ablation                                                                 |
| exp3  | OV-Freeze Control                                                             |
| exp4  | Baseline Comparison                                                           |
| exp5  | Cross-Dataset + LDP Privacy-Utility Trade-off                                 |
| exp6  | Speculative Decoding Alpha Measurement                                        |
| exp7  | Privacy Evaluation (ASV-EER, GLO Attack)                                      |
| exp8  | Latency Benchmark                                                             |
| exp9  | CoT Ablation                                                                  |
| exp10 | Teacher Scale Selection                                                       |
| exp11 | Quantisation Scheme Comparison                                                |
| exp12 | FraudFusion Baseline                                                          |
| exp13 | Fusion Strategy (multimodal fusion ablation)                                  |
| exp14 | Multi-model same-data comparison (BF16 transformers vs Q4_K_M GGUF llama.cpp) |

## Configuration

Edit `config/experiments.yaml` to adjust data sources, model paths, experiment parameters, etc.

### Data Source Modes

Three data source modes are supported:

- `auto`: Prefer real data, fall back to synthetic when unavailable
- `taf28k`: Force real TAF-28k data, raise error if unavailable
- `synthetic`: Force synthetic data

### Data Loading Priority

`load_fraud_texts()` follows this priority chain:

1. **Local files** — Config-specified paths (TAF-28k, ChiFraud, AdvFraud-3k)
2. **HuggingFace datasets** — `JimmyMa99/TeleAntiFraud` (requires internet)
3. **Backend data_loader** — Legacy backend (requires cluster environment)
4. **AdvFraud-3k** — Pre-built adversarial dataset (`data/AdvFraud3k/advfraud3k.json`)
5. **Simple random fallback** — Minimal random text generator (no templates)

### Environment Variables

| Variable                  | Description                                          | Default        |
| ------------------------- | ---------------------------------------------------- | -------------- |
| `REALEVAL_DATA_ROOT`    | Root directory for data files                        | `./data/`    |
| `REALEVAL_DATA__SOURCE` | Data source mode (`auto`/`taf28k`/`synthetic`) | `auto`       |
| `REALEVAL_OUTPUT_ROOT`  | Output directory for results                         | `./outputs/` |

## Datasets

### AdvFraud-3k (Adversarial Fraud Dataset)

A 3,000-sample adversarial Chinese fraud text dataset for robustness evaluation.

The dataset is pre-built and available in `data/AdvFraud3k/advfraud3k.json` (compiled format in `advfraud_3k_compiled.json`).
It contains 1,000 fraud samples adapted from TAF-28k with 8 adversarial perturbation strategies
(synonym, syntactic, dialect, metaphor, tone, key info, splitting, cross-domain) plus 2,000
novel fraud templates.

> **Note:** The build script (`data/scripts/build_advfraud3k.py`) has been replaced by the
> pre-compiled dataset. To regenerate, use `advfraud_3k_compiled.json` as the source.

## Cluster Deployment

Run on RunPod H100:

```bash
./cluster/setup_runpod.sh
./cluster/launch_runpod_h100.sh
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run HF dataset integration tests
pytest tests/test_hf_data.py -v
```

## Outputs

Experiment results are saved under `outputs/results/`, including:

- Per-experiment JSON result files
- `summary.csv`: Summary table
- `tables.md`: Paper-format tables
- `figures/`: Data figures (PNG + PDF)

## H100 Paper Validation (one command)

```bash
bash run_h100.sh                    # paper-grade (real Qwen + H100)
bash run_h100.sh --smoke            # sandbox verification
bash run_h100.sh --distributed      # 8x H100 via torchrun + NCCL
python -m experiments.runner --exp 1,3,6    # run specific experiments
```

Pipeline: CUDA check → GPU detect → env report → model load → benchmark → metrics → save.
Deliverables in `outputs/results/`: metrics.json, latency.csv, throughput.csv, memory.csv, paper_table.md,
and paper_tables/{table1_main,table2_ablation,table3_efficiency}.tex. Config overlay: config/h100.yaml.
