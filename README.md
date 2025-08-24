# Honeywell Multivariate Anomaly Detection (HMVA)

A production-ready pipeline + Streamlit dashboard for **multivariate time-series anomaly detection** on Tennessee Eastman Process–style datasets.
Learns a **baseline "normal" window**, scores every timestamp **0–100**, and explains anomalies with **`top_feature_1..7`** per row.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Dataset & Assumptions](#dataset--assumptions)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
  * [A) Streamlit Dashboard (recommended)](#a-streamlit-dashboard-recommended)
  * [B) CLI Batch Scoring (optional)](#b-cli-batch-scoring-optional)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Pipeline Details](#pipeline-details)
* [Scoring, Calibration & Severity](#scoring-calibration--severity)
* [Explainability (Top Contributors)](#explainability-top-contributors)
* [Screenshots (what to capture)](#screenshots-what-to-capture)
* [Troubleshooting](#troubleshooting)
* [Performance Notes](#performance-notes)
* [Reproducibility](#reproducibility)
* [Deployment Options](#deployment-options)
* [FAQ](#faq)
* [References](#references)
* [License](#license)

---

## Overview

Industrial processes emit many correlated sensor & actuator signals. HMVA provides a reusable way to:

1. **Learn "normal"** from a baseline training window.
2. **Score anomalies 0–100** across the timeline.
3. Provide **per-row explainability** via `top_feature_1..7`.

The app ships with a **pretrained run on the TEP dataset** and lets a tester **upload new datasets with the same schema** to re-run analysis in-app.

---

## Key Features

* **End-to-end pipeline**: validate time index, impute gaps, drop constants, fit ensemble (PCA + IsolationForest + z-scores), score, calibrate, smooth.
* **Human-friendly scale**: percentile-scaled **0–100** anomaly score.
* **Calibration contract**: On baseline, **mean < 10** and **max < 25**.
* **Explainability**: `top_feature_1..7` per timestamp (most influential sensors).
* **Dashboard UX**:
  * **Tab 1**: Pretrained TEP results (auto-processes raw if scored CSV absent).
  * **Tab 2**: Upload **RAW** (re-process) or **Scored** (visualize immediately).
* **Portable**: Pure Python stack; runs on CPU; Docker-friendly.

---

## Dataset & Assumptions

**Input schema (rigid):**

* One timestamp column named **`Time`** (`YYYY-MM-DD HH:MM`).
* Remaining columns are **numeric** process variables (flows, temperatures, pressures, valve positions, compositions).
* **Column names must match** the provided TEP file (row count may vary).

**Windows (configurable defaults used in TEP run):**

* **Training (baseline):** `2004-01-01 00:00` → `2004-01-05 23:59`
* **Analysis end:** `2004-01-19 07:59`
* **Guardrail:** baseline must contain **≥ 72 rows**.

---

## Requirements

* **Python**: 3.11 (3.10–3.12 likely OK)
* **OS**: Windows/macOS/Linux
* **Dependencies** (installed via `requirements.txt`):
  * pandas, numpy, scipy, scikit-learn, streamlit, plotly, matplotlib, seaborn

> *Tip*: For frozen runs, pin exact versions (see **Reproducibility**).

---

## Installation

```powershell
# Windows PowerShell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### A) Streamlit Dashboard (recommended)

From project root:

```powershell
streamlit run dashboard.py
```

**Tab 1 — Original Dataset Analysis**

* Shows pretrained TEP results.
* If `TEP_Output_Scored.csv` is missing, the app **auto-processes** `TEP_Train_Test.csv` on first load (cached).

**Tab 2 — Custom Dataset Analysis**

* **Raw CSV (will be processed):** upload a CSV with **identical schema** → pipeline runs in-app → same visuals as Tab 1.
* **Scored CSV (already processed):** upload a CSV that already includes `Abnormality_score` + `top_feature_1..7` → instant visualization.

> Exports: scored CSV & severity summary CSV (download buttons in Tab 2).

---

### B) CLI Batch Scoring (optional)

From project root, create `TEP_Output_Scored.csv` in one shot:

**Single line:**

```powershell
python .\main.py --input_csv "TEP_Train_Test.csv" --output_csv "TEP_Output_Scored.csv" --time_col "Time" --train_start "2004-01-01 00:00" --train_end "2004-01-05 23:59" --analysis_end "2004-01-19 07:59" --smooth 3
```

**PowerShell multiline (use backticks):**

```powershell
python .\main.py `
  --input_csv "TEP_Train_Test.csv" `
  --output_csv "TEP_Output_Scored.csv" `
  --time_col "Time" `
  --train_start "2004-01-01 00:00" `
  --train_end   "2004-01-05 23:59" `
  --analysis_end "2004-01-19 07:59" `
  --smooth 3
```

---

## Configuration

| Parameter      | Where           | Default            | Notes                                   |
| -------------- | --------------- | ------------------ | --------------------------------------- |
| `time_col`     | CLI & Dashboard | `Time`             | Case-sensitive column name              |
| `train_start`  | CLI & Dashboard | `2004-01-01 00:00` | Start of baseline window                |
| `train_end`    | CLI & Dashboard | `2004-01-05 23:59` | End of baseline; **≥ 72 rows** required |
| `analysis_end` | CLI & Dashboard | `2004-01-19 07:59` | End of analysis/scoring window          |
| `smooth`       | CLI & Dashboard | `3`                | Odd integer; `1` disables smoothing     |

---

## Project Structure

```
HONEYWELL_ANOMALY_PROJECT/
├─ pipeline/
│  ├─ data.py        # load_csv, validate intervals, impute & clean, split windows
│  ├─ model.py       # AnomalyEnsemble (PCA + IsolationForest + z-scores)
│  ├─ scoring.py     # rank/percentile scaling, calibration, smoothing
│  └─ io.py          # add_output_columns (score + top_feature_1..7)
├─ dashboard.py      # Streamlit app (Tab 1 pretrained, Tab 2 uploads)
├─ main.py           # optional CLI runner (writes scored CSV)
├─ requirements.txt
└─ TEP_Train_Test.csv
```

> Note: No separate `postprocess.py` — reporting/visuals are embedded in the dashboard (and optional exports in Tab 2).

---

## Pipeline Details

1. **Load & Align** (`pipeline.data.load_csv`)
   * Parse `Time` → DateTimeIndex; sort; drop dupes; keep numeric columns.

2. **Validate Intervals** (`validate_regular_intervals`)
   * Check cadence; warn on gaps.

3. **Impute & Clean** (`impute_and_clean`)
   * Forward-fill then time interpolation for small gaps;
   * Drop constant / all-NaN columns.

4. **Split Windows** (`split_train_analysis`)
   * Produce `train_df` (baseline) & `analysis_df` (scored range);
   * Raise if `< 72` training rows.

5. **Train Ensemble** (`model.AnomalyEnsemble.fit`)
   * Standardize (z-scores), learn **PCA** normal subspace, fit **IsolationForest**.

6. **Score All Rows** (`AnomalyEnsemble.score_rows`)
   * Get per-row signals: z-score aggregates, PCA residuals, IF scores;
   * **rank_normalize → average → percentile_scale → 0–100**.

7. **Calibrate & Smooth** (`scoring.calibrate_training`, `smooth_series`)
   * Enforce: training **mean < 10**, **max < 25**; optional median smoothing.

8. **Explainability** (`io.add_output_columns`)
   * Compute per-feature contribution (|z| + |PCA recon error| + IF sensitivity);
   * Write **`top_feature_1..7`** (pad with `""` if fewer).

---

## Scoring, Calibration & Severity

* **Score range**: **0–100**, percentile-scaled.
* **Calibration** (on training window):
  * **mean < 10**, **max < 25** (re-scales if needed).
* **Severity bands**:
  * 0–10 **Normal**
  * 11–30 **Slight**
  * 31–60 **Moderate**
  * 61–90 **Significant**
  * 91–100 **Severe**

This contract makes results comparable across runs/datasets with the same schema.

---

## Explainability (Top Contributors)

For each timestamp, HMVA computes feature contributions:

```
contrib(feature) ≈ |z_score| + |PCA_reconstruction_error| + IF_sensitivity
```

Sort descending, keep the **top 7** names:

```
top_feature_1, …, top_feature_7
```

If fewer than 7 contributors remain after thresholds, the rest are padded with empty strings.

---

## Screenshots (what to capture)

Include these in your report/submission:

1. **Training Validation** (baseline mean < 10, max < 25)
2. **Anomaly Timeline** (with training period band + guide lines at 10/30/60/90)
3. **Severity Distribution & Stats**
4. **Top Anomalies Table** (`top_feature_1..7` visible)

Optional (nice extras): Feature contribution over time, anomaly incidents, cumulative severe curve, correlation shift.

---

## Troubleshooting

* **"Insufficient training data (<72 rows)."**
  Enlarge the training window in the sidebar/CLI so baseline has ≥ 72 rows.

* **"Time parse failed / column not found."**
  Ensure the CSV has a **`Time`** column (case-sensitive) in `YYYY-MM-DD HH:MM`.

* **Schema mismatch (missing/renamed variables).**
  RAW uploads must have **identical column names** to the TEP schema.
  Alternatively, upload a **scored CSV** that already follows the output contract.

* **Irregular intervals / big gaps.**
  Small gaps are interpolated; large gaps degrade quality—clean upstream or split the run.

---

## Performance Notes

* CPU-friendly; TEP-scale data processes in seconds–minutes.
* Complexity (n rows, d features):
  * Preprocess/Scoring: **O(n·d)**
  * PCA (fit on training): ≈ **O(n·d²)** when d ≪ n (common in TEP)
  * IsolationForest: ≈ **O(trees · n · log n)**

---

## Reproducibility

Pin versions in `requirements.txt` for frozen runs:

```
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.0
scipy==1.13.1
streamlit==1.36.0
plotly==5.24.1
matplotlib==3.9.2
seaborn==0.13.2
```

* Set `random_state` in PCA/IF if strict determinism is desired.
* Use a virtualenv (`.venv`) per the **Installation** step.

---

## Deployment Options

**Streamlit Community Cloud / Hugging Face Spaces**

* Push repo (include `dashboard.py`, `pipeline/`, `requirements.txt`, and optionally `TEP_Train_Test.csv`).
* App entry = `dashboard.py`.
* (Optional) add `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200
fileWatcherType = "poll"

[theme]
base = "dark"
```

**Docker**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

```bash
docker build -t hmva .
docker run -p 8501:8501 hmva
```

---

## FAQ

**Q: Can I upload any CSV?**
A: RAW uploads must have the **same columns** (names & types) as the provided TEP file. Row count can differ.

**Q: What if my training window is too short?**
A: The app will warn and stop; widen the window to ensure **≥ 72** points.

**Q: Do I need labels?**
A: No—this is **unsupervised** anomaly detection.

**Q: Where are the "top contributors"?**
A: In the output as `top_feature_1..7`, and visible in the dashboard's top anomalies table.

---

## References

* Downs & Vogel (1993) — TEP benchmark problem
* Jolliffe & Cadima (2016) — PCA review
* Jackson (2005) — PCA practice
* Wise & Gallagher (1996) — Chemometrics for process monitoring
* Montgomery (2009) — Statistical Quality Control (z-scores, baselines)
* Liu et al. (2008) — Isolation Forest
* pandas / NumPy / SciPy / scikit-learn / Streamlit / Plotly documentation

---

## License

This project uses an open-source Python stack. The Tennessee Eastman Process dataset is a widely used research benchmark; ensure your usage complies with your institution/competition terms. Code in this repository is provided for educational and evaluation purposes.

