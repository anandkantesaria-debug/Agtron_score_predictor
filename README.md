# ☕ Coffee Roast Color (Agtron) Predictor & Recommender

An end-to-end Machine Learning system that predicts coffee roast color (**Agtron score**) from roaster process parameters and provides an **inverse-optimization recommendation engine** to help roasters dial in roasting temperatures and quench volumes.

---

## 📌 Overview

During commercial coffee roasting, target color (Agtron commercial scale) dictates flavor development and product consistency. Dialing in batches typically requires trial-and-error roasts, resulting in lost coffee and machine downtime. 

This project delivers:
1. **Accurate Color Prediction:** A CatBoost regression model trained on over 140,000 roast profiles predicting final Agtron scores.
2. **Settings Recommendation Engine:** A vectorized what-if search that identifies optimal temperature and quench adjustments to bridge the gap to a target Agtron score.
3. **MLOps Lifecycle Management:** MLflow experiment tracking, model artifact logging, and automated champion-challenger model promotion on holdout evaluation data.
4. **Interactive UI:** A streamlined Streamlit interface designed for roastery operators.

---

## 🏗️ Architecture

```
                                  +-----------------------+
                                  | Data_time_moisture.csv|
                                  +-----------+-----------+
                                              |
                                              v
+------------------------+        +-----------+-----------+        +--------------------------+
|  MLflow Model Registry | <====> |   Catboost_train.py   | -----> | models/agtron_model.cbm  |
|  (Champion/Challenger) |        +-----------------------+        |     (Serving Model)      |
+------------------------+                                         +------------+-------------+
                                                                                |
                                                                                v
                                                                   +------------+-------------+
                                                                   |          app.py          |
                                                                   |     (Streamlit UI)       |
                                                                   +--------------------------+
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/your-username/Agtron_score_predictor.git
cd Agtron_score_predictor

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Training & Model Promotion
Make sure your MLflow tracking server is active (default: `http://127.0.0.1:5000` or specify `MLFLOW_TRACKING_URI`):
```bash
python Catboost_train.py
```
The script will:
- Train a new CatBoost model using validation-based early stopping.
- Evaluate the candidate model on holdout test data (`X_test`).
- Benchmark against the active MLflow **Champion** model on the same test split.
- Promote the candidate to Champion and update `models/agtron_model.cbm` if performance improves.

### 4. Running Tests
```bash
pytest
```

---

## 📊 Features & Model Details

- **Input Features:**
  - `Consumed` *(float)*: Consumed green/roasted weight in pounds.
  - `Temp` *(float)*: Roaster temperature in °F.
  - `Quench` *(float/int)*: Volume/weight of quench water applied in pounds.
  - `Month` *(categorical string)*: Roast month (seasonality adjustment).
  - `Blend` *(categorical string)*: Blend/recipe code.
- **Model:** `CatBoostRegressor` (`depth=6`, `eval_metric="RMSE"`, `early_stopping_rounds=50`).
- **Target:** `Agtron` color reading (typically 15–65; lower = darker roast).

---

## 🛠️ Project Structure

```
├── .devcontainer/         # VS Code devcontainer configuration
├── .github/workflows/     # CI/CD pipelines (GitHub Actions)
├── models/                # Serving model directory
│   └── agtron_model.cbm   # Promoted production CatBoost model
├── tests/                 # Unit and validation tests
│   └── test_model.py
├── .gitignore             # Git ignore rules for datasets and artifacts
├── app.py                 # Streamlit web application
├── Catboost_train.py      # Model training, MLflow evaluation & promotion
├── requirements.txt       # Production dependencies with pinned versions
├── verify_model.py        # Model verification smoke script
└── README.md
```

