import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

def remove_outliers(df, columns, z_thresh=3):
    z = np.abs(stats.zscore(df[columns]))
    return df[(z < z_thresh).all(axis=1)]

def label_encode(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

def binarize_target(df, target_column='SFR'):
    df[target_column] = df[target_column].apply(lambda x: 1 if x >= 6 else 0)
    return df

def engineer_features(df):
    # Step 1: Remove outliers
    df = remove_outliers(df, ['Payload (kg)', 'Launch Cost ($M)'])

    # Step 2: Encode categorical features
    categorical_cols = ['Launch Class', 'Orbit Altitude', 'Tech Type', 'Description']
    df = label_encode(df, categorical_cols)

    # Step 3: Binarize SFR
    df = binarize_target(df, 'SFR')

    return df
