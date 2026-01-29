import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


#  Generate synthetic 3D data
np.random.seed(42)  # For reproducibility
n_samples = 100
X = np.random.randn(n_samples, 3)
true_beta = np.array([2.0, -1.0, 1.5])
y = X @ true_beta + np.random.normal(0, 0.5, size=n_samples)

#Center the data
X_mean = X.mean(axis=0)
y_mean = y.mean()
X_c = X - X_mean
y_c = y - y_mean


#Covariance and eigen-decomposition

cov = np.cov(X_c.T)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Top principal component (PC1)
PC1 = eigvecs[:, 0].reshape(-1, 1)
Z1 = X_c @ PC1  # Project data onto PC1

# PCR: Regress y on PC1
reg = LinearRegression()
reg.fit(Z1, y_c)
gamma = reg.coef_[0]
intercept_pcr = reg.intercept_

# Back-project to original space
beta_pcr = PC1.flatten() * gamma
y_pcr = Z1.flatten() * gamma + intercept_pcr

# OLS Regression 

ols = LinearRegression()
ols.fit(X_c, y_c)
y_ols = ols.predict(X_c)

# Visualization

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter original centered data
ax.scatter(X_c[:, 0], X_c[:, 1], X_c[:, 2], color='blue', alpha=0.6, label='Centered Data')

# PC1 vector
scale = 3
ax.quiver(0, 0, 0, PC1[0]*scale, PC1[1]*scale, PC1[2]*scale,
          color='green', linewidth=2, label='PC1 Direction')

# Projected points onto PC1
proj_points = Z1 @ PC1.T
ax.scatter(proj_points[:, 0], proj_points[:, 1], proj_points[:, 2],
           color='red', s=15, label='Projected Points (PC1)')

# Projection lines for the first 10
for i in range(10):
    ax.plot([X_c[i, 0], proj_points[i, 0]],
            [X_c[i, 1], proj_points[i, 1]],
            [X_c[i, 2], proj_points[i, 2]],
            'k--', linewidth=0.7)

# PCR regression line
line_range = np.linspace(Z1.min(), Z1.max(), 100).reshape(-1, 1)
line_pts = line_range @ PC1.T
ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
        color='magenta', linewidth=3, label='PCR Regression (PC1)')

# OLS Plane
x_range = np.linspace(X_c[:, 0].min(), X_c[:, 0].max(), 10)
y_range = np.linspace(X_c[:, 1].min(), X_c[:, 1].max(), 10)
xx, yy = np.meshgrid(x_range, y_range)
zz = (ols.intercept_ +
      ols.coef_[0] * xx +
      ols.coef_[1] * yy +
      ols.coef_[2] * np.zeros_like(yy))

# Plot the plane (no label for direct legend assignment)
ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')


#  Legend + Labels
legend_elements = [
    Line2D([0], [0], color='blue', marker='o', linestyle='', label='Centered Data'),
    Line2D([0], [0], color='green', lw=2, label='PC1 Direction'),
    Line2D([0], [0], color='red', marker='o', linestyle='', label='Projected Points (PC1)'),
    Line2D([0], [0], color='magenta', lw=2, label='PCR Regression (PC1)'),
    Patch(facecolor='cyan', edgecolor='k', label='OLS Plane')
]

ax.set_xlabel('X1 (centered)')
ax.set_ylabel('X2 (centered)')
ax.set_zlabel('X3 (centered)')
ax.set_title('Principal Component Regression vs OLS (3D Visualization)')
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.show()

