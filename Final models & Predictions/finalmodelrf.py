from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import time
import numpy as np

X_train= np.loadtxt('X_imputed.txt')
y_train= np.loadtxt('y.txt')

# Default hyperparameters
start_time = time.time()
reg = RandomForestRegressor()
model = reg.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)

# Save trained models
joblib.dump(model,'FinalModelRf.pkl')