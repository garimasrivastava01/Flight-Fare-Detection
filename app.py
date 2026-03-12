"""
╔══════════════════════════════════════════════════════╗
║     FLIGHT FARE DETECTION - FLASK WEB APPLICATION    ║
║     Run: python app.py                               ║
║     Open: http://localhost:5000                      ║
╚══════════════════════════════════════════════════════╝
"""

from flask import Flask, render_template, request, jsonify
import pickle, json, numpy as np, os

app = Flask(__name__)

# ── Load model & assets ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, 'model', 'model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

with open(os.path.join(BASE, 'model', 'encodings.json')) as f:
    ENC = json.load(f)

with open(os.path.join(BASE, 'model', 'metrics.json')) as f:
    METRICS = json.load(f)

with open(os.path.join(BASE, 'model', 'stats.json')) as f:
    STATS = json.load(f)

FEATURE_ORDER = ['airline', 'source_city', 'departure_time', 'stops',
                 'arrival_time', 'destination_city', 'class',
                 'duration', 'days_left', 'booking_window', 'is_peak']

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', metrics=METRICS, stats=STATS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Encode categoricals
        def enc(col, val):
            return ENC[col].get(str(val), 0)

        days_left = int(data['days_left'])
        duration  = float(data['duration'])
        dep_time  = data['departure_time']

        booking_window = (3 if days_left <= 7 else
                          2 if days_left <= 14 else
                          1 if days_left <= 30 else 0)
        is_peak = 1 if dep_time in ['Early_Morning', 'Morning'] else 0

        features = np.array([[
            enc('airline',          data['airline']),
            enc('source_city',      data['source_city']),
            enc('departure_time',   dep_time),
            enc('stops',            data['stops']),
            enc('arrival_time',     data['arrival_time']),
            enc('destination_city', data['destination_city']),
            enc('class',            data['class']),
            duration,
            days_left,
            booking_window,
            is_peak,
        ]])

        price = float(MODEL.predict(features)[0])
        price = max(1105, min(123071, price))

        # Confidence interval from individual trees
        tree_preds = np.array([t.predict(features)[0] for t in MODEL.estimators_[:, 0]])
        ci_lo = float(np.percentile(tree_preds, 5))
        ci_hi = float(np.percentile(tree_preds, 95))

        category = ('Budget' if price < 8000 else
                    'Mid-Range' if price < 25000 else 'Premium')

        return jsonify({
            'success': True,
            'price': round(price, 2),
            'ci_lo': round(max(1105, ci_lo), 2),
            'ci_hi': round(min(123071, ci_hi), 2),
            'category': category,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
def stats():
    return jsonify(STATS)


@app.route('/metrics')
def metrics():
    return jsonify(METRICS)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  ✈  FLIGHT FARE DETECTION — FLASK APP")
    print("  🌐  Open browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
