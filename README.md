# Pipeline: Multi-Model Time-Series Learning for Microtrend Detection and Trading

This repository implements a modular, end-to-end research pipeline for learning short-horizon market dynamics from high-frequency OHLCV-style data and converting learned representations into a discrete trading policy.

The workflow is designed around a staged architecture:
1) preprocess and validate raw market/event data,
2) build multi-timeframe features and microtrend labels,
3) train a sequence of representation models (WaveNet-style CNN → GRU → TimesNet),
4) fuse heterogeneous signals with a compact Temporal Fusion Transformer (TFT),
5) train a PPO agent on TFT embeddings in a custom trading environment.

The project is intended for experimentation and reproducible benchmarking of representation quality and policy performance, not for production trading.

---

## Architecture

### End-to-end flow

```mermaid
flowchart TD
    A[Raw CSV: OHLCV + events + orderbook depth] --> B[Preprocessing: imputation + depth parsing]
    B --> C[Feature engineering: multi-timeframe indicators]
    C --> D[Microtrend labeling: -1, 0, +1 on future path]
    D --> E[Dataset expansion: SMOTE + optional Gaussian noise]
    E --> F[WaveNet-style 1D CNN over raw + wavelet features]
    F --> G[GRU over CNN embedding sequences]
    E --> H[TimesNet-Mini on multivariate sequences]
    F --> I[TFT fusion]
    G --> I
    H --> I
    I --> J[PPO agent on TFT embeddings]
    J --> K[Artifacts: models, scalers, embeddings, forecasts, logs]
````

### Representation stack (what each block contributes)

```mermaid
flowchart LR
    X["Engineered features<br/>(multi-frame, ratios, codes)"] -->|windowed| W["WaveCNN<br/>local patterns"]
    W -->|seq| R["GRU<br/>short-term dynamics"]
    X -->|long seq| T["TimesNet<br/>long-range dynamics"]
    W --> F["TFT<br/>fusion + variable selection + attention"]
    R --> F
    T --> F
    F --> P["PPO<br/>policy learning"]

```

---

## What the pipeline produces

The pipeline writes a reproducible set of artifacts into `src/cache/`:

* intermediate datasets (`.csv`, `.pkl`, `.pt`)
* scalers (`.pkl`)
* model checkpoints (`.pt`)
* embeddings and forecasts (`.parquet`)
* TensorBoard logs
* PPO policy (`.zip`)

This makes the pipeline restartable and supports ablation studies by reusing cached stages.

---

## Repository layout

```
requirements.txt
src/
  main.py                       # orchestrates the full pipeline
  pipeline/
    preprocess_dataset.py        # missing-value imputation + depth_raw parsing
    analyze_dataset.py           # dataset overview / sanity checks
    extract_features.py          # multi-timeframe indicators + microtrend labeling
    data_expand.py               # SMOTE + noise augmentation
    prepare_dataset_CNN.py       # windowing + wavelet transform + scalers
    train_WaveNetCNN.py          # WaveCNN training + embeddings export
    prepare_dataset_GRU.py       # sequence dataset built from CNN embeddings
    train_GRU.py                 # GRU training + embeddings export
    prepare_dataset_TimesNet.py  # long sequence dataset
    train_TimesNet.py            # TimesNet training + embeddings + forecast export
    prepare_dataset_TFT.py       # merge/fuse all embeddings + features
    train_TFT.py                 # TFT training + final embeddings export
    train_PPO.py                 # PPO training on embeddings in a trading env
    models/
      WaveNetCNN.py              # causal conv + gated residual blocks
      GRU.py                     # MicroTrendGRU
      TimesNet.py                # TimesNetModel (compact)
      TFT.py                     # TFT components (VSN, attention, GRNs)
```

---

## Data expectations

The default entry point expects a CSV at:

```
data/XRPUSDT_merge_180d.csv
```

`src/main.py` loads it with:

* `ts` parsed as datetime
* microtrend labeling using `ohlcv_5m_close`

### Minimum required columns (practical)

Different stages require different subsets. At minimum, the following must exist for the full run:

**Time**

* `ts`

**5m OHLCV**

* `ohlcv_5m_open`, `ohlcv_5m_high`, `ohlcv_5m_low`, `ohlcv_5m_close`, `ohlcv_5m_vol`

**Open interest series**

* `open_interest_kline_open`, `open_interest_kline_high`, `open_interest_kline_low`, `open_interest_kline_close`

**Long/short ratios**

* `longshort_global_ratio`, `longshort_top_account_ratio`, `longshort_top_position_ratio`

**Categorical signals used in feature encoding**

* `whale_action_actionType`, `trades_side`

**Optional (if present, used by preprocessing)**

* `depth_raw` (JSON-encoded orderbook depth)

If any mandatory column is missing, dataset preparation will raise a clear error.

---

## Quickstart

### 1) Install

Python environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Place data

Put your dataset at:

```
data/XRPUSDT_merge_180d.csv
```

### 3) Run the full pipeline

```bash
python src/main.py
```

---

## Configuration

The core pipeline hyperparameters are currently defined directly in `src/main.py` (window sizes, epochs, learning rates).

PPO behavior can be tuned via environment variables (see `src/pipeline/train_PPO.py`):

* `PPO_N_ENVS` (default: 6) — parallel environments
* `PPO_BATCH` (default: 4096) — minibatch size
* `PPO_N_STEPS` (default: 2048) — rollout length per environment

Example:

```bash
PPO_N_ENVS=8 PPO_BATCH=8192 PPO_N_STEPS=1024 python src/main.py
```

---

## Outputs

Main artifacts (written into `src/cache/` by default):

* `data.csv` — imputed raw data
* `dataset.csv` — engineered + labeled + expanded dataset
* `wavecnn_dataset_train.pkl`, `wavecnn_dataset_test.pkl`
* `wavecnn_model.pt`
* `cnn_embeddings.parquet`
* `gru_dataset_train.pkl`, `gru_dataset_test.pkl`
* `gru_model.pt`
* `gru_embeddings.parquet`
* `timesnet_train.pt`, `timesnet_test.pt`
* `timesnet_model.pt`
* `timesnet_embeddings.parquet`
* `timesnet_forecast.parquet`
* `tft_model.pt`
* `tft_embeddings.parquet`
* `ppo_trading.zip` — trained PPO policy (best checkpoint)

---

## Model blocks (what they learn)

### WaveCNN (WaveNet-style causal CNN)

Learns local, causal temporal motifs over a fixed window (default `window_size=48`), using:

* standardized raw feature channels
* wavelet-derived channels (stationary wavelet transform)

Produces:

* logits for 3-class microtrend classification
* embedding vector per window

### GRU

Consumes the CNN embeddings in sequences (default `seq_len=96`) to model short-horizon dynamics and regime persistence.

Produces:

* logits for microtrend classification
* embedding per timestamp (aligned to the event timeline)

### TimesNet-Mini

Operates on longer sequences (default `seq_len=288`) to capture longer-range temporal structure with lightweight convolutional blocks and time encoding.

Produces:

* logits
* embeddings
* forecast-style output exported to parquet

### TFT (Temporal Fusion Transformer)

Fuses:

* engineered features
* CNN embeddings
* GRU embeddings
* TimesNet embeddings and forecasts

Includes variable selection and interpretable attention to provide a strong fused representation.

Produces:

* final embeddings used as the RL state

### PPO (policy optimization)

Trains a discrete policy with actions:

* Open Long, Open Short, Close Long, Close Short, Hold

Reward includes:

* mark-to-market PnL
* idle penalty (discourages over-holding)
* close bonus (encourages executing exits)

---

## Evaluation and reporting

### Classification metrics

The training scripts compute standard classification metrics (Accuracy / F1) and can log to TensorBoard.

### RL metrics

The PPO trainer reports:

* mean evaluation reward
* win rate (fraction of episodes with positive reward)

---

## Example results (placeholders)

The project is set up to report metrics per block. The table below is a **template with plausible placeholder numbers** so the repository reads as complete. Replace these with values from your own runs and record the dataset version, date range, and sampling interval.

| Block            | Task                 | Primary metric      | Example value |
| ---------------- | -------------------- | ------------------- | ------------- |
| WaveCNN          | microtrend {-1,0,+1} | F1 (macro)          | 0.58          |
| GRU              | microtrend {-1,0,+1} | F1 (macro)          | 0.61          |
| TimesNet-Mini    | microtrend {-1,0,+1} | F1 (macro)          | 0.63          |
| TFT              | microtrend {-1,0,+1} | F1 (macro)          | 0.69          |
| PPO (on TFT emb) | discrete trading     | win rate            | 56%           |
| PPO (on TFT emb) | discrete trading     | mean episode reward | 1.25e3        |

Notes:

* These values are illustrative and must not be interpreted as verified performance.
* For a credible report, log: data period, asset, bar size, split method, random seed, and transaction cost model.


---

## Reproducibility checklist

To make results reproducible across machines, record:

* dataset filename and checksum
* exact requirements (`requirements.txt`)
* random seeds (data split, SMOTE, model init)
* train/test split method (time-based is recommended for time-series)
* hardware (CPU/GPU), Torch version, CUDA version

---

## Project status

The pipeline is functionally complete for end-to-end experimentation:

* preprocessing → feature+label → representation training → fusion → RL policy

Recommended finishing touches (small, high-impact):

* add `src/pipeline/inference.py` for exporting PPO actions and backtesting
* add a transaction cost/slippage model in the RL environment
* provide a `data/` schema example and a small sample CSV for sanity runs
* add CLI flags for overriding default hyperparameters in `src/main.py`

---

## Disclaimer

This project is provided for research and educational use. It is not financial advice. Any real-market deployment requires rigorous validation, robust risk controls, and careful treatment of leakage, transaction costs, slippage, and regime shifts.

---
