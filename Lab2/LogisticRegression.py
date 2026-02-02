import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# Generate synthetic data
X, y = make_blobs(n_samples=200, 
                  centers=2, # two classes
                  n_features=2,
                  cluster_std=2.5,
                  random_state=42) 

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Create a mesh grid for plotting decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200)
)

# Predict class probabilities on the grid
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] # Probability of the positive class
Z = Z.reshape(xx.shape) # reshape to match grid shape 200x200

# Plotting
plt.figure(figsize=(10, 6))
# Plot decision boundary and margins
plt.contourf(xx, yy, Z, levels=25, cmap="RdBu", alpha=0.6)
# Plot original data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap="RdBu")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Probability of Positive Class")
plt.show()

