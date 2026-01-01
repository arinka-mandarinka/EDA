import pandas as pd
from catboost import CatBoostClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import joblib

data = pd.read_csv("preprocessed_data.csv")

X = data.drop("Revenue", axis=1)
y = data["Revenue"]

FEATURES = joblib.load("features.joblib")
X = X[FEATURES]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

model = CatBoostClassifier()
model.load_model("trained_model.cbm")

explainer = ClassifierExplainer(
    model,
    X_test,
    y_test,
    labels=["No Revenue", "Revenue"]
)

db = ExplainerDashboard(
    explainer,
    title="Online Shoppers Purchasing Intention Dashboard",
    whatif=False,
    shap_interaction=False,
    decision_trees=False,
    importances=True,
    model_summary=True,
    contributions=True,
    shap_dependence=True
)

db.to_yaml(
    "dashboard.yaml",
    explainerfile="explainer.joblib",
    dump_explainer=True
)

print(" Дашборд успешно создан!")
