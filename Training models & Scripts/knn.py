from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV, KFold
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler

def RMSE(pred, test):
    return np.sqrt(np.mean((pred - test)**2))

X = np.loadtxt('X_imputed.txt')
y = np.loadtxt('y.txt')

# Outer evaluation split
outer = KFold(n_splits=5, shuffle=False)

# Inner evaluation split
inner = KFold(n_splits=4, shuffle=False)

# Standarization of the data
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

## Defining the hyperparameters space
param_KNN = {
    'n_neighbors': list(range(5, 50, 2)),  
    'metric': ['euclidean', 'manhattan', 'hamming']  
}

## Initializing the dictionaries for the trained models
KNN_def = {
    "Models": [],
    "Scores": [],
     "Times": []}

KNN_rand = {
    "Models": [],
    "Scores": [], "Times": []}
KNN_halving = {
    "Models": [],
    "Scores": [], "Times": []}

# Training the models with nested cross-validation
# Default model
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X_sc[train_index], X_sc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    KNN_reg = KNeighborsRegressor()
    model = KNN_reg.fit(X_train, y_train)
    score = RMSE(model.predict(X_test), y_test)
    KNN_def["Models"].append(model)
    KNN_def["Scores"].append(score)
    print(f"Iteration {i+1} of default model completed.")

end_time = time.time()
KNN_def["Times"].append(end_time - start_time)
print(KNN_def["Times"])

# Random Search
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X_sc[train_index], X_sc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    KNN_reg = KNeighborsRegressor()
    rand_search_KNN = RandomizedSearchCV(KNN_reg, n_iter=20, cv=inner, param_distributions=param_KNN,
                                        scoring='neg_root_mean_squared_error', random_state=100512068)
    rand_search_KNN.fit(X_train, y_train)
    model = rand_search_KNN.best_estimator_
    score = RMSE(model.predict(X_test), y_test)
    KNN_rand["Models"].append(model)
    KNN_rand["Scores"].append(score)
    print(f"Iteration {i+1} of Random Search model completed.")

end_time = time.time()
KNN_rand["Times"].append(end_time - start_time)
print(KNN_rand["Times"])

start_time = time.time()
# Halving Grid Search
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X_sc[train_index], X_sc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    KNN_reg = KNeighborsRegressor()
    halving_search_KNN = HalvingGridSearchCV(KNN_reg, param_KNN, cv=inner, min_resources="exhaust",
                                            max_resources="auto", scoring='neg_root_mean_squared_error',
                                            random_state=100512068)
    halving_search_KNN.fit(X_train, y_train)
    model = halving_search_KNN.best_estimator_
    score = RMSE(model.predict(X_test), y_test)
    KNN_halving["Models"].append(model)
    KNN_halving["Scores"].append(score)

    print(f"Iteration {i+1} of Halving Grid Search model completed.")
end_time = time.time()
KNN_halving["Times"].append(end_time - start_time)
print(KNN_halving["Times"])

# Save trained models
KNN = {"KNN_def": KNN_def, "KNN_rand": KNN_rand, "KNN_halving": KNN_halving}
joblib.dump(KNN, 'KNN_models.pkl')
