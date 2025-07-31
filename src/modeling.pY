from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def train_models(df):
    X = df.drop(columns=['SFR'])
    y = df['SFR']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'min_samples_split': [2, 4],
        'max_depth': [4, 6],
        'min_samples_leaf': [2, 4],
        'random_state': [0, 42]
    }

    rfc = RandomForestClassifier()
    rfc_grid = GridSearchCV(rfc, param_grid, cv=5, verbose=0, n_jobs=-1)
    rfc_grid.fit(X_train, y_train)
    best_rfc = rfc_grid.best_estimator_

    dtc = DecisionTreeClassifier()
    dtc_grid = GridSearchCV(dtc, param_grid, cv=5, verbose=0, n_jobs=-1)
    dtc_grid.fit(X_train, y_train)
    best_dtc = dtc_grid.best_estimator_

    best_rfc.fit(X_train, y_train)
    best_dtc.fit(X_train, y_train)

    return best_rfc, best_dtc, X_test, y_test
