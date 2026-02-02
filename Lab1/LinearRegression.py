import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(23) 
m = 100
x_feature =3*np.random.rand(m, 1)
y = 3 + 5 * x_feature + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), x_feature]

# Plotting dataset
plt.scatter(x_feature, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Theta found by normal equation:", theta_normal)
# The output shoud be very close to the groundtruth of 3(3.1) and 5(4.8)


# Using sklearn linear model: 
model = LinearRegression(fit_intercept=False)
model.fit(X_b, y)
theta_sklearn = model.coef_.reshape(-1, 1)

# print("Theta found by scikit-learn:", model.coef_.flatten())
print("Theta found by scikit-learn:", theta_sklearn)


# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(x_feature, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

model_2 = LinearRegression(fit_intercept=True)
model_2.fit(X_train, y_train)
print("Theta found by scikit-learn:", model_2.coef_.flatten())

y_pred = model_2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

