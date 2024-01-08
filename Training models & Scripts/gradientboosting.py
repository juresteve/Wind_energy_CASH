from sklearn.ensemble import HistGradientBoostingRegressor
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

# Defining the hyperparameters space
param_gb = {
    'learning_rate': list(np.arange(0.01, 0.1, 0.005)),  
    'min_samples_leaf': [10, 30, 50],  
    "max_depth": range(5, 26, 2)
}

# Initializing the dictionaries for the trained models
gb_def = {"Models": [], "Scores": [], "Time": []}
gb_rand = {"Models": [], "Scores": [], "Time": []}
gb_halving = {"Models": [], "Scores": [], "Time": []}

# Training the models with nested cross-validation

# Default hyperparameters
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gb_reg = HistGradientBoostingRegressor()
    model = gb_reg.fit(X_train, y_train)
    score = RMSE(model.predict(X_test), y_test)
    gb_def["Models"].append(model)
    gb_def["Scores"].append(score)
    print(f"Iteration {i+1} of default model completed.")

end_time = time.time()
gb_def["Time"].append(end_time - start_time)

# Random Search
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gb_reg = HistGradientBoostingRegressor()
    rand_search_gb = RandomizedSearchCV(gb_reg, n_iter=20, cv=inner, param_distributions=param_gb,
                                        scoring='neg_root_mean_squared_error', random_state=100512068)
    rand_search_gb.fit(X_train, y_train)
    model = rand_search_gb.best_estimator_
    score = RMSE(model.predict(X_test), y_test)
    gb_rand["Models"].append(model)
    gb_rand["Scores"].append(score)
    print(f"Iteration {i+1} of Random Search model completed.")

end_time = time.time()
gb_rand["Time"].append(end_time - start_time)

start_time = time.time()
# Halving Grid Search
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gb_reg = HistGradientBoostingRegressor()
    halving_search_gb = HalvingGridSearchCV(gb_reg, param_gb, cv=inner, min_resources="exhaust",
                                            max_resources="auto", scoring='neg_root_mean_squared_error',
                                            random_state=100512068)
    halving_search_gb.fit(X_train, y_train)
    model = halving_search_gb.best_estimator_
    score = RMSE(model.predict(X_test), y_test)
    gb_halving["Models"].append(model)
    gb_halving["Scores"].append(score)

    print(f"Iteration {i+1} of Halving Grid Search model completed.")
end_time = time.time()
gb_halving["Time"].append(end_time - start_time)

# Save trained models
gb = {"gb_def": gb_def, "gb_rand": gb_rand, "gb_halving": gb_halving}
joblib.dump(gb, 'GradientBoosting_models_julia.pkl')
