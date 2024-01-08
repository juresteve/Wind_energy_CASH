from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import numpy as np
import joblib
import time
import numpy as np

X_train= np.loadtxt('X_imputed.txt')
y_train= np.loadtxt('y.txt')

# Inner evaluation split
inner = KFold(n_splits=4, shuffle=False)

## Defining the hyperparameters space
param_trees = {
    "max_depth": range(10, 30, 1),
    "min_samples_split": range(2, 40, 1)}

# Random Search
start_time = time.time()
trees_reg = DecisionTreeRegressor()
rand_search_trees = RandomizedSearchCV(trees_reg, n_iter=20, cv=inner, param_distributions=param_trees,
                                        scoring='neg_root_mean_squared_error', random_state=100512068)
rand_search_trees.fit(X_train, y_train)
model = rand_search_trees.best_estimator_

end_time = time.time()
print(end_time - start_time)

# Save trained models
joblib.dump(model,'FinalModelTrees.pkl')