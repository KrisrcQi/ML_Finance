import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Creating variables and functions 
np.random.seed(23) 
m = 100
x_feature =3*np.random.rand(m, 1)
y = 3 + 5 * x_feature + np.random.randn(m, 1) # function
X_b = np.c_[np.ones((m, 1)), x_feature]

# Plotting oberservation
plt.scatter(x_feature, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Computing theta_0 and theta_1
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Theta found by normal equation:", theta_normal)
"""
You should have a pair of values that is very close to original function parameters (3, 5)

Output:
Theta found by normal equation: 
[[3.10749931]
[4.89149406]]

So, it's correct.
"""

# Estimate the parameters by linear regression:
model = LinearRegression(fit_intercept=False)
model.fit(X_b, y)
theta_sklearn = model.coef_.reshape(-1, 1)
# Print("Theta found by scikit-learn:", model.coef_.flatten())
print("Theta found by scikit-learn:", theta_sklearn)

""" output:
Theta found by scikit-learn: [[3.10749931]
 [4.89149406]]
"""


# Training a linear model to predict the parameters:
## Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(x_feature, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

## Building model
model_2 = LinearRegression(fit_intercept=True)
model_2.fit(X_train, y_train)
print("Theta found by scikit-learn:", model_2.coef_.flatten())

## Prediction and evaluation of the model
y_pred = model_2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)



