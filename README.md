# ✈ Flight Fare Detection — Flask Web App

> Professional ML-powered web application with real Kaggle data

---

## 📁 Project Structure

```
Flight_Fare_Flask/
│
├── app.py                  ← Flask backend (MAIN SERVER)
├── train_model.py          ← Run once to train & save model
├── requirements.txt        ← Python dependencies
│
├── model/                  ← Auto-created after training
│   ├── model.pkl
│   ├── encodings.json
│   ├── metrics.json
│   └── stats.json
│
├── templates/
│   └── index.html          ← Frontend HTML
│
└── static/
    ├── css/style.css       ← Styles
    └── js/main.js          ← JavaScript
```

---

## 🚀 Setup & Run (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place dataset
Put `Clean_Dataset.csv` in the **same folder** as `app.py`

### Step 3 — Train the model (run ONCE)
```bash
python train_model.py
```
This creates the `model/` folder with trained model files.

### Step 4 — Start the web app
```bash
python app.py
```

### Step 5 — Open browser
```
http://localhost:5000
```

---

## 🌐 API Endpoints

| Endpoint   | Method | Description              |
|------------|--------|--------------------------|
| `/`        | GET    | Main web interface       |
| `/predict` | POST   | Predict fare (JSON)      |
| `/stats`   | GET    | Dataset statistics       |
| `/metrics` | GET    | Model performance stats  |

### Sample `/predict` Request:
```json
POST /predict
{
  "airline": "Indigo",
  "source_city": "Delhi",
  "destination_city": "Mumbai",
  "departure_time": "Morning",
  "arrival_time": "Afternoon",
  "stops": "zero",
  "class": "Economy",
  "duration": 2.5,
  "days_left": 15
}
```

### Response:
```json
{
  "success": true,
  "price": 5420.0,
  "ci_lo": 4800.0,
  "ci_hi": 6200.0,
  "category": "Budget"
}
```

---

## 📊 Model Performance

| Metric     | Value              |
|------------|--------------------|
| Algorithm  | Gradient Boosting  |
| R² Score   | 97.23%             |
| MAE        | ₹2,166             |
| RMSE       | ₹3,780             |
| Training   | 3,00,153 records   |
| Dataset    | Kaggle Real Data   |
