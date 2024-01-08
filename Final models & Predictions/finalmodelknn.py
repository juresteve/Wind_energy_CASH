from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import numpy as np
import joblib
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train= np.loadtxt('X_imputed.txt')
y_train= np.loadtxt('y.txt')

# Inner evaluation split
inner = KFold(n_splits=4, shuffle=False)

# Standarization of the data
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)

## Defining the hyperparameters space
param_KNN = {
    'n_neighbors': list(range(5, 50, 2)),  
    'metric': ['euclidean', 'manhattan', 'hamming']  
}

# Random Search
start_time = time.time()
reg = KNeighborsRegressor()
rand_search = RandomizedSearchCV(reg, n_iter=20, cv=inner, param_distributions=param_KNN,
                                        scoring='neg_root_mean_squared_error', random_state=100512068)
rand_search.fit(X_train, y_train)
model = rand_search.best_estimator_

end_time = time.time()
print(end_time - start_time)

# Save trained models
joblib.dump(model,'FinalModelKnn.pkl')