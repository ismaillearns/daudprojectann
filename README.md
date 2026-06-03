# 🩺 Diabetes Prediction App — Streamlit

An ANN-based web app that predicts the likelihood of diabetes using the PIMA Indians Diabetes Dataset.

---

## 📁 Folder Structure

```
diabetes-app/
├── app.py               ← Streamlit web application
├── run_training.py      ← Script to train model & save artifacts
├── requirements.txt     ← Python dependencies
├── diabetes.csv         ← Dataset (you must add this)
├── diabetes_model.h5    ← Generated after training
└── scaler.pkl           ← Generated after training
```

---

## 🚀 Quick Start (Local)

### Step 1 — Add the dataset
Place your `diabetes.csv` file in this folder.  
Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model
```bash
python run_training.py
```
This creates `diabetes_model.h5` and `scaler.pkl`.

### Step 4 — Run the app
```bash
streamlit run app.py
```
Open your browser at http://localhost:8501

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this entire folder to a **GitHub repository**
2. Also upload `diabetes_model.h5` and `scaler.pkl` to the same repo
3. Go to https://share.streamlit.io
4. Sign in with GitHub → click **New app**
5. Select your repo, branch (`main`), and set **Main file path** to `app.py`
6. Click **Deploy** ✅

Your app will be live at:
`https://<your-github-username>-<repo-name>.streamlit.app`

---

## 📊 Model Details

| Parameter     | Value                    |
|---------------|--------------------------|
| Architecture  | ANN (2 hidden layers)    |
| Hidden layers | Dense(12, ReLU), Dense(8, ReLU) |
| Output        | Dense(1, Sigmoid)        |
| Optimizer     | Adam                     |
| Loss          | Binary Cross-Entropy     |
| Epochs        | 100                      |
| Accuracy      | ~0.82                    |

---

## 🛠 Tech Stack
- Python · TensorFlow/Keras · Scikit-learn · Streamlit · Pandas · NumPy
