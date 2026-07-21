import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "linear_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Minimalist & Modern HTML/CSS Template with Purple Shades
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-purple: #6D28D9;
            --primary-hover: #5B21B6;
            --purple-light: #DDD6FE;
            --purple-bg: #F3E8FF;
            --dark-text: #1F2937;
            --card-bg: #FFFFFF;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #2E1065 0%, #4C1D95 50%, #6D28D9 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 480px;
            padding: 32px;
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .header h1 {
            color: var(--primary-purple);
            font-size: 26px;
            font-weight: 700;
        }

        .header p {
            color: #6B7280;
            font-size: 14px;
            margin-top: 6px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-weight: 600;
            color: var(--dark-text);
            margin-bottom: 8px;
            font-size: 14px;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid var(--purple-light);
            border-radius: 8px;
            font-size: 15px;
            outline: none;
            transition: all 0.2s ease;
        }

        input[type="text"]:focus {
            border-color: var(--primary-purple);
            box-shadow: 0 0 0 3px rgba(109, 40, 217, 0.2);
        }

        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #7C3AED, #6D28D9);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.1s ease;
        }

        button:hover {
            background: linear-gradient(135deg, #6D28D9, #5B21B6);
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            background-color: var(--purple-bg);
            border-left: 4px solid var(--primary-purple);
            border-radius: 8px;
            text-align: center;
        }

        .result-box h3 {
            color: var(--primary-purple);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .result-box p {
            color: #4C1D95;
            font-size: 24px;
            font-weight: 700;
            margin-top: 4px;
        }

        .error {
            color: #DC2626;
            background-color: #FEE2E2;
            border-left-color: #DC2626;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>House Price Predictor</h1>
        <p>Enter feature values (comma-separated)</p>
    </div>

    <form method="POST" action="/predict">
        <div class="form-group">
            <label for="features">Input Features</label>
            <input type="text" id="features" name="features" placeholder="e.g. 1500, 3, 2" value="{{ features if features else '' }}" required>
        </div>
        <button type="submit">Predict Price</button>
    </form>

    {% if prediction %}
    <div class="result-box">
        <h3>Estimated Value</h3>
        <p>${{ prediction }}</p>
    </div>
    {% endif %}

    {% if error %}
    <div class="result-box error">
        <h3>Error</h3>
        <p style="font-size: 14px;">{{ error }}</p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            error="Model 'linear_model.pkl' not found on server."
        )

    try:
        raw_input = request.form.get("features", "")
        # Parse comma-separated inputs into floats
        feature_list = [float(x.strip()) for x in raw_input.split(",") if x.strip()]
        
        # Reshape for single prediction sample
        features_array = np.array([feature_list])
        
        # Predict
        prediction_value = model.predict(features_array)[0]
        formatted_prediction = f"{prediction_value:,.2f}"

        return render_template_string(
            HTML_TEMPLATE, 
            prediction=formatted_prediction, 
            features=raw_input
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            error=f"Invalid Input: {str(e)}", 
            features=request.form.get("features", "")
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
