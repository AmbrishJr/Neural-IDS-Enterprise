# neural-ids-enterprise

Code for *An Integrated Neural Network Framework for Advanced Enterprise Network Intrusion Detection*.
It compares the neural architectures discussed in the review (Table 2) against classical
baselines on public intrusion detection datasets. Evaluation uses the metrics the review recommends:
macro-F1, MCC, FPR, ROC-AUC, PR-AUC, and per-flow latency.

## Status

| Week | Work | Status |
|---|---|---|
| 1 | Data pipeline, Experiment 1 on UNSW-NB15: classical baselines, MLP, CNN-1D, BiLSTM, CNN-LSTM, ensemble, autoencoder | done |
| 2 | Zero-day (leave-one-attack-out), cross-dataset (CIC-IDS2017 to CSE-CIC-IDS2018), more datasets | planned |
| 3 | Adversarial robustness (FGSM/PGD, adversarial training), SHAP explanations, FastAPI + Streamlit demo | planned |
| 4 | Final figures and tables, paper Methodology and Results sections | planned |

## Setup

```bash
pip install -r requirements.txt
```

The dataset is already in `data/raw/unsw_nb15/`. It comes from the Hugging Face mirror
`wwydmanski/UNSW-NB15`, a Kaggle copy (`dhoogla/unswnb15`) of the official UNSW-NB15 partitions.
To download it again:

```bash
mkdir -p data/raw/unsw_nb15 && cd data/raw/unsw_nb15
for f in train.parquet test.parquet; do
  curl -sL -o $f https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/$f
done
```

The mirror names the files the opposite way from the official partitions. Its `test.parquet`
(175,341 flows) is the official **training** set, and its `train.parquet` (82,332 flows) is the
official **testing** set. `src/ids/data.py` uses the official definition. The mirror also drops some
columns (`sttl`, `dttl`, and several `ct_*` counters) that are known to leak the label in this dataset.

UNSW-NB15 is free for academic research. Cite Moustafa &amp; Slay (2015).

## Run

```bash
python scripts/run_experiment1.py            # binary + 10-class, about 25 min on an 8 GB Apple-silicon laptop
python scripts/run_experiment1.py --task binary
```

Outputs:
- `results/exp1_binary.csv`, `results/exp1_multiclass.csv`: one row per model, all metrics
- `results/exp1_multiclass_per_class_f1.csv`: per-attack-category F1 for every model
- `figures/exp1_model_comparison.png`, `figures/exp1_confusion_multiclass.png`
- `results/*.pt`: trained model weights

## Method (Experiment 1)

- **Split:** official train/test partitions. 15% of the training set is held out, stratified by attack
  category, for early stopping. The test set is never used for any choice during training.
- **Preprocessing** (fitted on the training split only): signed `log1p` and standardisation for the 29
  numeric features; one-hot encoding of `proto`, `service`, and `state` (top 12 values + "other");
  TCP sequence numbers dropped. This gives 65 input features.
- **Class imbalance:** class weights for all models (inverse square-root frequency for the neural nets).
- **Neural models:** AdamW, batch 512, at most 30 epochs, early stopping on validation macro-F1
  (patience 5). Sequence models read the feature vector as a sequence of length 65.
- **Autoencoder:** trained on Normal flows only. The threshold is the 95th percentile of
  reconstruction error on validation Normal flows. It never sees an attack label.
- **Ensemble:** average of predicted probabilities from GBDT, RandomForest, CNN-LSTM, and MLP.
- **Latency:** median CPU batch-inference time per flow (20k flows, 3 repeats).
- Seed 42 everywhere.

## Results: Experiment 1 (UNSW-NB15 official test set, 82,332 flows)

| Model | Binary acc | Binary macro-F1 | Binary FPR | 10-class macro-F1 | CPU latency (µs/flow) |
|---|---|---|---|---|---|
| Logistic regression | 0.801 | 0.795 | 0.306 | 0.345 | 0.1 |
| Random forest | 0.883 | 0.880 | 0.214 | **0.507** | 2.6 |
| GBDT (HistGB) | **0.905** | **0.903** | 0.145 | 0.465 | 5.7 |
| MLP | 0.859 | 0.853 | 0.261 | 0.466 | 1.4 |
| CNN-1D | 0.868 | 0.863 | 0.242 | 0.463 | 46.6 |
| BiLSTM | 0.843 | 0.834 | 0.311 | 0.434 | 54.8 |
| CNN-LSTM | 0.856 | 0.850 | 0.273 | 0.448 | 25.7 |
| Ensemble (GBDT+RF+CNN-LSTM+MLP) | 0.880 | 0.876 | 0.219 | 0.497 | 35.4 |
| Autoencoder (no attack labels) | 0.854 | 0.854 | **0.086** | n/a | 1.2 |

Findings:
1. On tabular flow features, tree ensembles beat every neural model. Sequence models (BiLSTM,
   CNN-LSTM) add 20-50x latency without improving accuracy, because a flow's feature vector has no real order.
2. The unsupervised autoencoder, trained on Normal traffic only, matches the supervised neural nets
   (macro-F1 0.854) and has the lowest false-positive rate (8.6%). This supports its use for zero-day detection.
3. The 10-class task is much harder (best macro-F1 0.51). Analysis, Backdoor, and DoS are nearly
   inseparable (per-class F1 < 0.45), and 25% of Normal flows are predicted as Fuzzers. These are
   known properties of UNSW-NB15. They also show why the "99% accuracy" figures common in the
   literature come from random re-splits rather than the official test partition.

## Layout

```
src/ids/data.py      loading + leakage-safe preprocessing
src/ids/models.py    MLP, CNN-1D, BiLSTM, CNN-LSTM, Autoencoder
src/ids/train.py     training loops (early stopping, class weights)
src/ids/metrics.py   macro-F1, MCC, FPR, ROC/PR-AUC, latency
scripts/             one runner per experiment
```

## License

Code: MIT (see `LICENSE`). The UNSW-NB15 dataset is not included and keeps its own academic-use terms.
