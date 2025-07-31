# Altima: Aerospace Launch Trends & Intelligent Market Analysis

Altima (Aerospace Launch Trends & Intelligent Market Analysis) is a data-driven machine learning project designed to analyze, classify, and predict the viability of aerospace launch companies using public domain SpaceFund Realty (SFR) ratings data. This project leverages modern data science tools to assess the landscape of commercial space ventures and provide strategic insights into their technological, financial, and operational viability.

## Project Objective

To develop an end-to-end machine learning pipeline that predicts the SFR rating class of a launch company based on features like payload capacity, launch cost, vehicle type, tech domain, and strategic market focus.

## Features

* Preprocessing pipeline to clean and normalize messy real-world data
* Exploratory data analysis with insightful visualizations
* Feature engineering including categorical encoding and derived labels
* Binary classification of SFR ratings (High vs Low viability)
* Model training using Random Forest and Decision Tree classifiers
* Evaluation with confusion matrix, classification report, and accuracy
* Feature importance insights for model interpretability
* Interactive Streamlit dashboard for live inference and file upload

## Technologies Used

* Python 3.11+
* Pandas, NumPy, Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Joblib (for model persistence)

## Directory Structure

```
Altima-Aerospace-Launch-Trends-Intelligent-Market-Analysis/
├── data/                   # Raw CSV input files
├── models/                 # Saved ML models (.pkl)
├── plots/                  # EDA visualizations
├── src/                    # Source modules
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
├── app.py                  # Streamlit dashboard
├── vain.py                 # Local training/testing script
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Altima-Aerospace-Launch-Trends-Intelligent-Market-Analysis.git
cd Altima-Aerospace-Launch-Trends-Intelligent-Market-Analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Use Case

Altima is designed for:

* Venture capital analysts evaluating space startups
* Aerospace strategy and R\&D teams
* Data scientists looking to apply ML in a high-stakes technical domain

## Example Output

* Classification accuracy: \~74%
* Interactive predictions from custom CSV uploads
* Visual feedback through confusion matrix and feature importance

---

*Note: This project is a research-driven analysis and does not constitute financial advice or commercial recommendation.*
