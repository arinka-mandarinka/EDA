import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data = pd.read_csv("preprocessed_data.csv")

    X = data.drop("Revenue", axis=1)
    y = data["Revenue"]

    FEATURES = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=42,
        stratify=y
    )

    cat_features = np.where(X_train.dtypes == object)[0]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test)
    )

    model.save_model("trained_model.cbm")
    joblib.dump(FEATURES, "features.joblib")

    print("Модель обучена.")
    print("Файл trained_model.cbm сохранён.")
    print("Файл features.joblib сохранён.")
