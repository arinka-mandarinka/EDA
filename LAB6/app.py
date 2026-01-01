from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
import joblib

app = Flask(__name__)

# Загружаем модель и список признаков
model = CatBoostClassifier()
model.load_model("trained_model.cbm")

FEATURES = joblib.load("features.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])


    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(str).str.lower().map(
            {"true": 1, "false": 0}
        )

    # Приводим порядок колонок
    df = df[FEATURES]

    prediction = model.predict_proba(df)[0, 1]

    return jsonify({"Revenue_Probability": float(prediction)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
