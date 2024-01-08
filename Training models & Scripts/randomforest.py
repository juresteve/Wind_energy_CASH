from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV, KFold
import numpy as np
import joblib
import time

def RMSE(pred, test):
    return np.sqrt(np.mean((pred - test)**2))

X = np.loadtxt('X_imputed.txt')
y = np.loadtxt('y.txt')

# Outer evaluation split
outer = KFold(n_splits=5, shuffle=False)

# Inner evaluation split
inner = KFold(n_splits=4, shuffle=False)

# Initializing the dictionaries for the trained models
rf_def = {"Models": [], "Scores": [], "Times": []}

# Training the models with nested cross-validation
# Default model
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf_reg = RandomForestRegressor()
    model = rf_reg.fit(X_train, y_train)
    score = RMSE(model.predict(X_test), y_test)
    rf_def["Models"].append(model)
    rf_def["Scores"].append(score)
    print(f"Iteration {i+1} of default model completed.")

end_time = time.time()
rf_def["Times"].append(end_time - start_time)

# Save trained models
rf = {"rf_def": rf_def}
joblib.dump(rf, 'rf_models.pkl')
