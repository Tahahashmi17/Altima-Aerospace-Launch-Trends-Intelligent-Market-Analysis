import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.data_preprocessing import (
    load_and_clean_data,
    standardize_categories,
    categorize_description,
    final_cleanup,
)
from src.feature_engineering import engineer_features

st.set_page_config(page_title="SFR Analysis Dashboard", layout="wide")
st.title("ALTIMA Analysis Dashboard")

@st.cache_resource
def load_models():
    rf = joblib.load("models/random_forest.pkl")
    dt = joblib.load("models/decision_tree.pkl")
    return {"Random Forest": rf, "Decision Tree": dt}

models = load_models()

uploaded_file = st.file_uploader("Upload launch data (.csv)", type=["csv"])

if uploaded_file:
    try:
        file_content = uploaded_file.getvalue()
        csv_data = io.StringIO(file_content.decode("utf-8"))

        df_raw = pd.read_csv(csv_data, encoding="utf-8", thousands=",")
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        csv_data.seek(0)
        df = load_and_clean_data(csv_data)
        df = standardize_categories(df)
        df = categorize_description(df)
        df = final_cleanup(df)
        df = engineer_features(df)

        st.subheader("Preprocessed Data")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Data Distribution")
        with st.expander("Show Distribution Charts"):
            fig, ax = plt.subplots(1, 3, figsize=(18, 4))
            sns.countplot(data=df, x="Launch Class", ax=ax[0], palette="Set2")
            sns.countplot(data=df, x="Tech Type", ax=ax[1], palette="Set2")
            sns.countplot(data=df, x="Description", ax=ax[2], palette="Set2")
            for a in ax:
                a.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        st.subheader("Choose model for prediction")
        selected_model_name = st.selectbox("Select a model", list(models.keys()))
        model = models[selected_model_name]

        if st.button("Predict SFR Category"):
            X = df.drop(columns=["SFR"], errors="ignore")
            y = df["SFR"] if "SFR" in df.columns else None

            predictions = model.predict(X)
            df["SFR_Prediction"] = predictions

            st.success(f"Prediction complete using {selected_model_name}")
            st.dataframe(df[["SFR_Prediction"] + list(X.columns)].head(), use_container_width=True)

            if y is not None:
                st.subheader("Evaluation Metrics")
                report = classification_report(y, predictions, output_dict=True)
                st.json(report)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y, predictions)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "High"], yticklabels=["Low", "High"])
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

            st.subheader("Feature Importance")
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=X.columns)
                importances = importances.sort_values(ascending=False)

                fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
                sns.barplot(x=importances, y=importances.index, palette="viridis", ax=ax_fi)
                ax_fi.set_title("Feature Importances")
                st.pyplot(fig_fi)
            else:
                st.info("❗Feature importance is not available for this model.")

    except Exception as e:
        st.error(f"Error during preprocessing or prediction:\n\n{str(e)}")

else:
    st.info("Please upload a `.csv` file to begin.")
