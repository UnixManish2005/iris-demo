"""Train and save the Iris species classification pipeline."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=250, random_state=42))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions, target_names=iris.target_names))
output = Path(__file__).parent / "iris_model.joblib"
joblib.dump(model, output)
print(f"Saved model to {output}")
