from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import time

def RMSE(pred, test):
    return np.sqrt(np.mean((pred - test)**2))

X = np.loadtxt('X_imputed_finaldata.txt')
y = np.loadtxt('y.txt')

# Standarization of the data
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Outer evaluation split
outer = KFold(n_splits=5, shuffle=False)

# Inner evaluation split
inner = KFold(n_splits=4, shuffle=False)

## Base models (estimators)
KNN = joblib.load('KNN_models.pkl')
Trees = joblib.load('dectrees_models.pkl')
RandForest = joblib.load('rf_models.pkl')
GradBoost = joblib.load('GradientBoosting_models.pkl')
ExtraTrees = joblib.load('extratrees1_models.pkl')

estimators = [
    ('KNN', KNN["KNN_rand"]["Models"][0]),
    ('Trees', Trees["trees_rand"]["Models"][0]),
    ('RandomForest', RandForest["rf_def"]["Models"][0]),
    ('GradientBoosting', GradBoost["gb_rand"]["Models"][0]),
    ('ExtraTrees', ExtraTrees["extra_def"]["Models"][0])
]

# Initializing the dictionary for the trained models
lr = {"Models": [], "Scores": [], "Time": []}

## Linear regression
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X_sc)):
    print(f"Iteration {i+1} of linear regression.")
    X_train, X_test = X_sc[train_index], X_sc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    meta_model_LR = LinearRegression()
    stacked_model_LR = StackingRegressor(estimators=estimators, final_estimator=meta_model_LR)
    model = stacked_model_LR.fit(X_train, y_train)
    score = RMSE(model.predict(X_test), y_test)
    lr["Models"].append(model)
    lr["Scores"].append(score)
    print(f"Iteration {i+1} completed.")

end_time = time.time()
lr["Time"].append(end_time - start_time)

# Save trained models
joblib.dump(lr, 'Stacking_models.pkl')
