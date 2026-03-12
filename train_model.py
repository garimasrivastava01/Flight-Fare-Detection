"""
╔══════════════════════════════════════════════════════╗
║     FLIGHT FARE DETECTION — MODEL TRAINING SCRIPT    ║
║     Run ONCE before starting the Flask app:          ║
║       python train_model.py                          ║
╚══════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle, json, os

print("\n" + "="*55)
print("  ✈  FLIGHT FARE — MODEL TRAINING")
print("="*55)

# ── 1. Load Data ──────────────────────────────────────────
print("\n📂 Loading dataset...")
DATA_PATH = 'Clean_Dataset.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n❌ '{DATA_PATH}' not found!\n"
        "   Download from: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction\n"
        "   Place 'Clean_Dataset.csv' in this folder."
    )

df = pd.read_csv(DATA_PATH)
df.drop(columns=['Unnamed: 0', 'flight'], inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)
print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── 2. Encode ─────────────────────────────────────────────
print("\n⚙️  Encoding categorical columns...")
cat_cols = ['airline', 'source_city', 'departure_time', 'stops',
            'arrival_time', 'destination_city', 'class']
le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
print("   ✅ Done")

# ── 3. Feature Engineering ────────────────────────────────
print("\n🛠️  Feature engineering...")
df['booking_window'] = pd.cut(df['days_left'],
                               bins=[0,7,14,30,49],
                               labels=[3,2,1,0]).astype(int)
df['is_peak'] = df['departure_time'].apply(lambda x: 1 if x in [1,4] else 0)
print("   ✅ Added: booking_window, is_peak")

# ── 4. Split ──────────────────────────────────────────────
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 5. Train ──────────────────────────────────────────────
print("\n🤖 Training Gradient Boosting model (may take 1-2 min)...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────
preds = model.predict(X_test)
r2   = round(r2_score(y_test, preds) * 100, 2)
mae  = round(mean_absolute_error(y_test, preds))
rmse = round(np.sqrt(mean_squared_error(y_test, preds)))

print(f"\n📊 Results:")
print(f"   R² Score : {r2}%")
print(f"   MAE      : ₹{mae:,}")
print(f"   RMSE     : ₹{rmse:,}")

# ── 7. Save ───────────────────────────────────────────────
os.makedirs('model', exist_ok=True)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/encodings.json', 'w') as f:
    json.dump(le_map, f)

with open('model/metrics.json', 'w') as f:
    json.dump({'r2': r2, 'mae': mae, 'rmse': rmse,
               'records': len(df), 'features': list(X.columns)}, f)

# Stats
orig = pd.read_csv(DATA_PATH)
stats = {
    'airline_avg': orig.groupby('airline')['price'].mean().round(0).astype(int).to_dict(),
    'stops_avg':   orig.groupby('stops')['price'].mean().round(0).astype(int).to_dict(),
    'class_avg':   orig.groupby('class')['price'].mean().round(0).astype(int).to_dict(),
    'city_avg':    orig.groupby('source_city')['price'].mean().round(0).astype(int).to_dict(),
}
with open('model/stats.json', 'w') as f:
    json.dump(stats, f)

print("\n✅ All files saved in model/ folder:")
print("   model/model.pkl")
print("   model/encodings.json")
print("   model/metrics.json")
print("   model/stats.json")
print("\n" + "="*55)
print("  🚀 Now run:  python app.py")
print("  🌐 Open:     http://localhost:5000")
print("="*55 + "\n")
