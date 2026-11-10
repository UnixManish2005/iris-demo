import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Iris AI | Species Predictor", page_icon="🌸", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f7f8ff 0%, #eef7f4 100%); }
.block-container { padding-top: 2rem; max-width: 1200px; }
.hero { padding: 1.7rem 2rem; border-radius: 22px; color: white;
        background: linear-gradient(110deg, #263c75, #5865b8 60%, #3b9c91);
        margin-bottom: 1.3rem; }
.hero h1 { color: white; font-size: 2.35rem; margin: 0; }
.hero p { color: #e9edff; margin: .45rem 0 0; font-size: 1.02rem; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e8eaf3;
    padding: 14px 16px; border-radius: 16px; box-shadow: 0 3px 12px #25335a0b; }
section[data-testid="stSidebar"] { background: #f0f2fb; }
</style>
<div class="hero">
  <h1>🌸 Iris AI — Species Predictor</h1>
  <p>Explore the classic Iris dataset and predict a flower species with a machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    path = Path(__file__).parent / "iris_model.joblib"
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
    if not path.exists():
        joblib.dump(model, path)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, iris, X_test, y_test, accuracy

model, iris, X_test, y_test, accuracy = load_model_and_data()
species = list(iris.target_names)
features = list(iris.feature_names)

with st.sidebar:
    st.header("🧭 Navigation")
    page = st.radio("Choose a page", ["Predict species", "Explore dataset", "Model performance"])
    st.divider()
    st.caption("Model: StandardScaler + Random Forest")
    st.caption("Dataset: scikit-learn Iris · 150 samples")

if page == "Predict species":
    st.subheader("Make a prediction")
    st.write("Enter the flower's sepal and petal measurements in centimetres.")
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("#### 🌿 Flower measurements")
        c1, c2 = st.columns(2)
        with c1:
            sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
            sw = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
        with c2:
            pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
            pw = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)
        predict = st.button("✨ Predict iris species", type="primary", use_container_width=True)
    with right:
        st.markdown("#### 🔎 Prediction result")
        if predict:
            sample = pd.DataFrame([[sl, sw, pl, pw]], columns=features)
            pred = int(model.predict(sample)[0])
            probs = model.predict_proba(sample)[0]
            st.success(f"Predicted species: **{species[pred].title()}**")
            st.metric("Model confidence", f"{probs[pred] * 100:.1f}%")
            prob_df = pd.DataFrame({"Species": [s.title() for s in species], "Probability": probs})
            fig = px.bar(prob_df, x="Probability", y="Species", orientation="h",
                         range_x=[0, 1], text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
                         color="Species", color_discrete_sequence=["#7586d8", "#55a99b", "#e5a45d"])
            fig.update_layout(height=260, showlegend=False, margin=dict(l=0,r=10,t=15,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Adjust the measurements, then click **Predict iris species**.")
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Iris_versicolor_3.jpg",
                     caption="Iris flower example", use_container_width=True)

elif page == "Explore dataset":
    st.subheader("Explore the Iris dataset")
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(species))).str.title()
    a, b, c = st.columns(3)
    a.metric("Total samples", len(df))
    b.metric("Features", len(features))
    c.metric("Species", len(species))
    st.markdown("#### Sample records")
    st.dataframe(df.drop(columns="target").head(10), use_container_width=True)
    st.markdown("#### Feature relationships")
    x_axis = st.selectbox("X-axis", features, index=2)
    y_axis = st.selectbox("Y-axis", features, index=3)
    fig = px.scatter(df, x=x_axis, y=y_axis, color="species",
                     color_discrete_sequence=["#7586d8", "#55a99b", "#e5a45d"],
                     hover_data=features, title="Iris measurements by species")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Feature distributions")
    selected = st.selectbox("Select a feature", features, key="distribution")
    fig2 = px.histogram(df, x=selected, color="species", barmode="overlay",
                        opacity=0.72, color_discrete_sequence=["#7586d8", "#55a99b", "#e5a45d"])
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.subheader("Model performance")
    y_pred = model.predict(X_test)
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{accuracy:.1%}")
    c2.metric("Test samples", len(y_test))
    c3.metric("Trees in forest", "250")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, x=[s.title() for s in species],
                    y=[s.title() for s in species], color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    report = classification_report(y_test, y_pred, target_names=species, output_dict=True)
    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.caption("Metrics use a stratified 80/20 train-test split with random_state=42.")
