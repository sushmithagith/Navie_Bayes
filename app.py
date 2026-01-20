import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Naive Bayes Classifier", layout="centered")
st.markdown("<h2 style='text-align:center;'>Predict output using Naive Bayes Algorithm</h2>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# Sidebar - CSV Upload
# -------------------------------------------------
st.sidebar.header("Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_count = X.shape[1]

else:
    # Demo dataset (numeric only)
    X, y = make_classification(
        n_samples=1000,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    feature_count = 3

# -------------------------------------------------
# Sidebar Inputs (Dynamic)
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Enter Feature Values")

input_values = []
for i in range(feature_count):
    value = st.sidebar.number_input(f"Feature {i+1}", value=0.0)
    input_values.append(value)

input_data = np.array([input_values])

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.markdown("---")
st.subheader("Prediction")

if st.button("Predict Output"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("Prediction: YES")
    else:
        st.error("Prediction: NO")

    st.write("Prediction Probability:")
    st.write(probability)

# -------------------------------------------------
# Accuracy
# -------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown("---")
st.subheader("Model Accuracy")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")
