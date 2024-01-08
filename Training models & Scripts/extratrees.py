from sklearn.ensemble import ExtraTreesRegressor
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

## Defining the hyperparameters space
param_trees = {
    "max_depth": range(10, 30, 1),
    "min_samples_split": range(2, 40, 1)}

## Initializing the dictionaries for the trained models
extra_def = {
    "Models": [],
    "Scores": [],
     "Times": []}
extra_rand = {
    "Models": [],
    "Scores": [], "Times": []}
extra_halving = {
    "Models": [],
    "Scores": [], "Times": []}
# Training the models with nested cross-validation
# Default model
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    extra_reg = ExtraTreesRegressor()
    model = extra_reg.fit(X_train, y_train)
    score = RMSE(model.predict(X_test), y_test)
    extra_def["Models"].append(model)
    extra_def["Scores"].append(score)
    print(f"Iteration {i+1} of default model completed.")

end_time = time.time()
extra_def["Times"].append(end_time - start_time)
print(extra_def["Times"])

# Random Search
start_time = time.time()
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    extra_reg = ExtraTreesRegressor()
    rand_search_trees = RandomizedSearchCV(extra_reg, n_iter=20, cv=inner, param_distributions=param_trees,
                                        scoring='neg_root_mean_squared_error', random_state=100512068)
    rand_search_trees.fit(X_train, y_train)
    model = rand_search_trees.best_estimator_
    score = RMSE(model.predict(X_test), y_test)
    extra_rand["Models"].append(model)
    extra_rand["Scores"].append(score)
    print(f"Iteration {i+1} of Random Search model completed.")

end_time = time.time()
extra_rand["Times"].append(end_time - start_time)
print(extra_rand["Times"])

start_time = time.time()
# Halving Grid Search
for i, (train_index, test_index) in enumerate(outer.split(X)):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    trees_reg = ExtraTreesRegressor()
    halving_search_trees = HalvingGridSearchCV(trees_reg, param_trees, cv=inner, min_resources="exhaust",
                                            max_resources="auto", scoring='neg_root_mean_squared_error',
                                            random_state=100512068)
    halving_search_trees.fit(X_train, y_train)
    model = halving_search_trees.best_estimator_
    score = RMSE(model.predict(X_test), y_test)
    extra_halving["Models"].append(model)
    extra_halving["Scores"].append(score)

    print(f"Iteration {i+1} of Halving Grid Search model completed.")
end_time = time.time()
extra_halving["Times"].append(end_time - start_time)
print(extra_halving["Times"])

# Save trained models
extra = {"extra_def": extra_def, "extra_rand": extra_rand, "extra_halving": extra_halving}
joblib.dump(extra, 'extratrees1_models.pkl')

