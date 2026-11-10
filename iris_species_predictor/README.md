# Iris AI — Species Predictor

A beginner-friendly machine-learning project that predicts **Iris setosa**, **Iris versicolor**, or **Iris virginica** from four flower measurements. Includes a Streamlit dashboard.

## Features
- Random Forest classifier in a scikit-learn preprocessing pipeline
- Interactive prediction sliders and class probabilities
- Dataset preview, scatter plot, and feature distributions
- Test accuracy, confusion matrix, and classification report
- Reproducible stratified 80/20 train-test split

## Run locally

Python 3.9+ is recommended.

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save the model (optional; the app can train it on first run)
python train_model.py

# 4. Launch the UI
streamlit run app.py
```



## Dataset
Uses the built-in Iris dataset shipped with scikit-learn: 150 samples, four numeric measurements (sepal length/width and petal length/width), and three species classes.

## Project structure
```text
iris_species_predictor/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Notes
- Measurements are in centimetres.
- The displayed test score is based on a held-out split; it is not a guarantee of performance on every future sample.
- Only load `iris_model.joblib` files you created or trust.
