import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import t, chi2
import os

# Create output directory if it doesn't exist
output_dir = 'output_basic_stats'
os.makedirs(output_dir, exist_ok=True)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# 1. MEASURES OF CENTRAL TENDENCY

def mean(data):
    return sum(data) / len(data)

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    return sorted_data[n//2]

def mode(data):
    freq = {}
    for val in data:
        freq[val] = freq.get(val, 0) + 1
    max_freq = max(freq.values())
    return [key for key, val in freq.items() if val == max_freq]

# Example data with increased sample size
np.random.seed(123)
data1 = np.random.normal(25, 8, 100).tolist()  # Increased to 100
print(f"Data (first 10): {data1[:10]}...")
print(f"Mean: {mean(data1):.2f}")
print(f"Median: {median(data1):.2f}")
print(f"Mode: {mode(data1)}")  # Note: Mode may not be meaningful for continuous data

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].hist(data1, bins=12, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(mean(data1), color='red', linestyle='--', label=f'Mean: {mean(data1):.2f}')
axes[0,0].axvline(median(data1), color='green', linestyle='--', label=f'Median: {median(data1):.2f}')
axes[0,0].set_title('Distribution with Central Tendency')
axes[0,0].legend()

axes[0,1].boxplot(data1, vert=True)
axes[0,1].set_title('Box Plot')
axes[0,1].set_ylabel('Values')

skewed_data = np.random.exponential(10, 100).tolist()  # Increased to 100
axes[1,0].hist(skewed_data, bins=12, alpha=0.7, color='orange', edgecolor='black')
axes[1,0].axvline(mean(skewed_data), color='red', linestyle='--', label=f'Mean: {mean(skewed_data):.2f}')
axes[1,0].axvline(median(skewed_data), color='green', linestyle='--', label=f'Median: {median(skewed_data):.2f}')
axes[1,0].set_title('Skewed Distribution')
axes[1,0].legend()

normal_data = np.random.normal(25, 5, 100).tolist()  # Increased to 100
axes[1,1].hist(normal_data, bins=12, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,1].axvline(mean(normal_data), color='red', linestyle='--', label=f'Mean: {mean(normal_data):.2f}')
axes[1,1].axvline(median(normal_data), color='green', linestyle='--', label=f'Median: {median(normal_data):.2f}')
axes[1,1].set_title('Normal-like Distribution')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'central_tendency.png'))
plt.show()

# 2. MEASURES OF VARIABILITY

def variance(data, sample=True):
    m = mean(data)
    squared_diff = [(x - m)**2 for x in data]
    divisor = len(data) - 1 if sample else len(data)
    return sum(squared_diff) / divisor

def std_deviation(data, sample=True):
    return math.sqrt(variance(data, sample))

def range_calc(data):
    return max(data) - min(data)

def quartiles(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1 = sorted_data[n//4]
    q2 = median(sorted_data)
    q3 = sorted_data[3*n//4]
    return q1, q2, q3

def iqr(data):
    q1, q2, q3 = quartiles(data)
    return q3 - q1

print(f"\nVariability Measures for data1:")
print(f"Variance (sample): {variance(data1):.2f}")
print(f"Standard Deviation: {std_deviation(data1):.2f}")
print(f"Range: {range_calc(data1):.2f}")
print(f"IQR: {iqr(data1):.2f}")

# Visualization of variability
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

low_var = np.random.normal(25, 2, 100).tolist()  # Increased to 100
high_var = np.random.normal(25, 10, 100).tolist()  # Increased to 100

axes[0,0].hist(low_var, bins=12, alpha=0.7, color='lightblue', label=f'Low Var: {variance(low_var):.2f}')
axes[0,0].axvline(mean(low_var), color='red', linestyle='--')
axes[0,0].set_title('Low Variability')
axes[0,0].legend()

axes[0,1].hist(high_var, bins=12, alpha=0.7, color='lightcoral', label=f'High Var: {variance(high_var):.2f}')
axes[0,1].axvline(mean(high_var), color='red', linestyle='--')
axes[0,1].set_title('High Variability')
axes[0,1].legend()

axes[1,0].boxplot([low_var, high_var], labels=['Low Var', 'High Var'])
axes[1,0].set_title('Box Plot Comparison')

x = np.linspace(mean(data1) - 3*std_deviation(data1), mean(data1) + 3*std_deviation(data1), 100)
y = np.exp(-0.5 * ((x - mean(data1)) / std_deviation(data1))**2) / (std_deviation(data1) * np.sqrt(2 * np.pi))
axes[1,1].plot(x, y, 'b-', linewidth=2)
axes[1,1].axvline(mean(data1), color='red', linestyle='--', label='Mean')
axes[1,1].axvline(mean(data1) + std_deviation(data1), color='orange', linestyle='--', label='+1 SD')
axes[1,1].axvline(mean(data1) - std_deviation(data1), color='orange', linestyle='--', label='-1 SD')
axes[1,1].set_title('Standard Deviation Visualization')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'variability.png'))
plt.show()

# 3. STANDARD ERROR

def standard_error(data):
    return std_deviation(data) / math.sqrt(len(data))

print(f"\nStandard Error: {standard_error(data1):.3f}")

# Visualization of standard error
sample_sizes = [10, 20, 50, 100, 200, 500]
se_values = []
population = np.random.normal(500, 100, 2000).tolist()  # Increased population size

for n in sample_sizes:
    sample = population[:n]
    se_values.append(standard_error(sample))

fig = plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, se_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Sample Size')
plt.ylabel('Standard Error')
plt.title('Standard Error vs Sample Size')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'standard_error.png'))
plt.show()

# 4. PROBABILITY DISTRIBUTIONS

def binomial_prob(n, k, p):
    def factorial(x):
        if x <= 1:
            return 1
        return x * factorial(x - 1)
    return (factorial(n) / (factorial(k) * factorial(n - k))) * (p**k) * ((1-p)**(n-k))

def poisson_prob(lam, k):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma)**2)

# Distribution visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

n, p = 20, 0.3
x_binom = list(range(0, n+1))
y_binom = [binomial_prob(n, k, p) for k in x_binom]
axes[0,0].bar(x_binom, y_binom, alpha=0.7, color='skyblue')
axes[0,0].set_title(f'Binomial Distribution (n={n}, p={p})')
axes[0,0].set_xlabel('k')
axes[0,0].set_ylabel('Probability')

lam = 3
x_poisson = list(range(0, 15))
y_poisson = [poisson_prob(lam, k) for k in x_poisson]
axes[0,1].bar(x_poisson, y_poisson, alpha=0.7, color='lightgreen')
axes[0,1].set_title(f'Poisson Distribution (λ={lam})')
axes[0,1].set_xlabel('k')
axes[0,1].set_ylabel('Probability')

mu, sigma = 0, 1
x_norm = np.linspace(-4, 4, 100)
y_norm = [normal_pdf(x, mu, sigma) for x in x_norm]
axes[1,0].plot(x_norm, y_norm, 'b-', linewidth=2)
axes[1,0].fill_between(x_norm, y_norm, alpha=0.3)
axes[1,0].set_title(f'Normal Distribution (μ={mu}, σ={sigma})')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('Density')

for mu_i, sigma_i in [(0, 1), (0, 2), (2, 1)]:
    y_norm_i = [normal_pdf(x, mu_i, sigma_i) for x in x_norm]
    axes[1,1].plot(x_norm, y_norm_i, linewidth=2, label=f'μ={mu_i}, σ={sigma_i}')
axes[1,1].set_title('Multiple Normal Distributions')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'probability_distributions.png'))
plt.show()

# 5. T-TEST

def t_test_one_sample(data, mu0):
    n = len(data)
    sample_mean = mean(data)
    sample_std = std_deviation(data)
    t_stat = (sample_mean - mu0) / (sample_std / math.sqrt(n))
    df = n - 1
    return t_stat, df

def t_test_two_sample(data1, data2, equal_var=True):
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = mean(data1), mean(data2)
    var1, var2 = variance(data1), variance(data2)
    
    if equal_var:
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
        se = math.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        se = math.sqrt(var1/n1 + var2/n2)
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    t_stat = (mean1 - mean2) / se
    return t_stat, df

# Example t-tests with increased sample sizes
group1 = np.random.normal(30, 5, 50).tolist()  # Increased to 50
group2 = np.random.normal(32, 5, 50).tolist()  # Increased to 50

t_stat1, df1 = t_test_one_sample(group1, 30)
t_stat2, df2 = t_test_two_sample(group1, group2)

print(f"\nOne-sample t-test (H0: μ = 30):")
print(f"t-statistic: {t_stat1:.3f}, df: {df1}")
print(f"\nTwo-sample t-test:")
print(f"t-statistic: {t_stat2:.3f}, df: {df2:.1f}")

# Visualization of t-distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.linspace(-4, 4, 100)
for df_val in [3, 5, 10, 30]:
    y = t.pdf(x, df_val)
    label = f'df={df_val}' if df_val < 30 else f'df={df_val} (≈Normal)'
    axes[0].plot(x, y, linewidth=2, label=label)

axes[0].set_title('t-Distribution with Different df')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

x_crit = np.linspace(-4, 4, 100)
y_crit = t.pdf(x_crit, df1)
axes[1].plot(x_crit, y_crit, 'b-', linewidth=2)
axes[1].fill_between(x_crit[x_crit <= -1.96], t.pdf(x_crit[x_crit <= -1.96], df1), alpha=0.3, color='red', label='Critical Region (α=0.05)')
axes[1].fill_between(x_crit[x_crit >= 1.96], t.pdf(x_crit[x_crit >= 1.96], df1), alpha=0.3, color='red')
axes[1].axvline(t_stat1, color='orange', linestyle='--', label=f't-stat: {t_stat1:.2f}')
axes[1].set_title('Critical Regions and Test Statistic')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 't_test.png'))
plt.show()

# 6. CHI-SQUARE TEST

def chi_square_test(observed, expected=None):
    if expected is None:
        expected = [sum(observed) / len(observed)] * len(observed)
    
    chi_stat = sum((o - e)**2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return chi_stat, df

def chi_square_independence(table):
    rows, cols = len(table), len(table[0])
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    grand_total = sum(row_totals)
    
    expected = []
    for i in range(rows):
        expected_row = []
        for j in range(cols):
            expected_freq = (row_totals[i] * col_totals[j]) / grand_total
            expected_row.append(expected_freq)
        expected.append(expected_row)
    
    chi_stat = 0
    for i in range(rows):
        for j in range(cols):
            chi_stat += (table[i][j] - expected[i][j])**2 / expected[i][j]
    
    df = (rows - 1) * (cols - 1)
    return chi_stat, df, expected

# Example chi-square tests with increased frequencies
observed_freq = [200, 240, 280, 240]  # Increased to ensure expected ≥5
expected_freq = [240, 240, 240, 240]
chi_stat, df = chi_square_test(observed_freq, expected_freq)

print(f"\nChi-square goodness of fit test:")
print(f"Chi-square statistic: {chi_stat:.3f}, df: {df}")
print(f"Expected frequencies: {[f'{e:.2f}' for e in expected_freq]}")

# Contingency table example with increased counts
contingency_table = [
    [120, 180, 240],
    [240, 300, 360],
    [180, 240, 300]
]

chi_stat_ind, df_ind, expected_table = chi_square_independence(contingency_table)
print(f"\nChi-square independence test:")
print(f"Chi-square statistic: {chi_stat_ind:.3f}, df: {df_ind}")
print(f"Expected frequencies:\n{[[f'{e:.2f}' for e in row] for row in expected_table]}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_chi = np.linspace(0, 20, 100)
for df_val in [1, 3, 5, 10]:
    y_chi = chi2.pdf(x_chi, df_val)
    axes[0].plot(x_chi, y_chi, linewidth=2, label=f'df={df_val}')

axes[0].set_title('Chi-square Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

im = axes[1].imshow(contingency_table, cmap='Blues', aspect='auto')
axes[1].set_title('Contingency Table Heatmap')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')

for i in range(len(contingency_table)):
    for j in range(len(contingency_table[0])):
        axes[1].text(j, i, contingency_table[i][j], ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chi_square.png'))
plt.show()

# 7. COVARIANCE AND CORRELATION

def covariance(x, y, sample=True):
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    
    mean_x, mean_y = mean(x), mean(y)
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    divisor = len(x) - 1 if sample else len(x)
    return cov / divisor

def correlation(x, y):
    cov_xy = covariance(x, y)
    std_x = std_deviation(x)
    std_y = std_deviation(y)
    if std_x * std_y == 0:
        return 0
    return cov_xy / (std_x * std_y)

# Example data with increased sample size
np.random.seed(123)
x_data = np.random.normal(5, 2, 100).tolist()  # Increased to 100
y_data = [1.5 * x + np.random.normal(0, 1) for x in x_data]
y_data2 = [-1.5 * x + np.random.normal(0, 1) for x in x_data]
y_data3 = np.random.normal(5, 2, 100).tolist()

cov_xy = covariance(x_data, y_data)
corr_xy = correlation(x_data, y_data)
corr_xy2 = correlation(x_data, y_data2)
corr_xy3 = correlation(x_data, y_data3)

print(f"\nCovariance (x, y): {cov_xy:.3f}")
print(f"Correlation (x, y): {corr_xy:.3f}")
print(f"Correlation (x, y2): {corr_xy2:.3f}")
print(f"Correlation (x, y3): {corr_xy3:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].scatter(x_data, y_data, alpha=0.7, color='blue', s=50)
axes[0,0].set_title(f'Positive Correlation (r = {corr_xy:.2f})')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')

axes[0,1].scatter(x_data, y_data2, alpha=0.7, color='red', s=50)
axes[0,1].set_title(f'Negative Correlation (r = {corr_xy2:.2f})')
axes[0,1].set_xlabel('X')
axes[0,1].set_ylabel('Y')

axes[1,0].scatter(x_data, y_data3, alpha=0.7, color='green', s=50)
axes[1,0].set_title(f'Weak Correlation (r = {corr_xy3:.2f})')
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Y')

corr_matrix = [
    [1.0, corr_xy, corr_xy2],
    [corr_xy, 1.0, corr_xy3],
    [corr_xy2, corr_xy3, 1.0]
]
im = axes[1,1].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
axes[1,1].set_title('Correlation Matrix')
axes[1,1].set_xticks([0, 1, 2])
axes[1,1].set_yticks([0, 1, 2])
axes[1,1].set_xticklabels(['X', 'Y1', 'Y2'])
axes[1,1].set_yticklabels(['X', 'Y1', 'Y2'])

for i in range(3):
    for j in range(3):
        axes[1,1].text(j, i, f'{corr_matrix[i][j]:.2f}', ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation.png'))
plt.show()

# 8. REGRESSION ANALYSIS

def simple_linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(xi**2 for xi in x)
    
    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    beta0 = (sum_y - beta1 * sum_x) / n
    return beta0, beta1

def predict(x, beta0, beta1):
    return beta0 + beta1 * x

def r_squared(y_actual, y_predicted):
    ss_res = sum((y_actual[i] - y_predicted[i])**2 for i in range(len(y_actual)))
    ss_tot = sum((y_actual[i] - mean(y_actual))**2 for i in range(len(y_actual)))
    return 1 - (ss_res / ss_tot)

def residuals(y_actual, y_predicted):
    return [y_actual[i] - y_predicted[i] for i in range(len(y_actual))]

# Example regression with increased sample size
np.random.seed(123)
x_reg = np.random.normal(5, 2, 100).tolist()
y_reg = [2 + 1.8 * x + np.random.normal(0, 1) for x in x_reg]

beta0, beta1 = simple_linear_regression(x_reg, y_reg)
y_pred = [predict(xi, beta0, beta1) for xi in x_reg]
r2 = r_squared(y_reg, y_pred)
resid = residuals(y_reg, y_pred)

print(f"\nSimple Linear Regression:")
print(f"Intercept (β0): {beta0:.3f}")
print(f"Slope (β1): {beta1:.3f}")
print(f"R-squared: {r2:.3f}")
print(f"Equation: y = {beta0:.3f} + {beta1:.3f}x")

# Multiple regression
def multiple_regression_2var(x1, x2, y):
    n = len(x1)
    X = [[1, x1[i], x2[i]] for i in range(n)]
    sum_x1 = sum(x1)
    sum_x2 = sum(x2)
    sum_y = sum(y)
    sum_x1_2 = sum(xi**2 for xi in x1)
    sum_x2_2 = sum(xi**2 for xi in x2)
    sum_x1_x2 = sum(x1[i] * x2[i] for i in range(n))
    sum_x1_y = sum(x1[i] * y[i] for i in range(n))
    sum_x2_y = sum(x2[i] * y[i] for i in range(n))
    
    beta1 = (sum_x1_y - sum_x1 * sum_y / n) / (sum_x1_2 - sum_x1**2 / n)
    beta2 = (sum_x2_y - sum_x2 * sum_y / n) / (sum_x2_2 - sum_x2**2 / n)
    beta0 = (sum_y - beta1 * sum_x1 - beta2 * sum_x2) / n
    return beta0, beta1, beta2

# Example multiple regression
x1_mult = np.random.normal(5, 2, 100).tolist()
x2_mult = np.random.normal(10, 3, 100).tolist()
y_mult = [3 + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 1) for x1, x2 in zip(x1_mult, x2_mult)]

beta0_mult, beta1_mult, beta2_mult = multiple_regression_2var(x1_mult, x2_mult, y_mult)
print(f"\nMultiple Regression:")
print(f"Intercept (β0): {beta0_mult:.3f}")
print(f"Coefficient x1 (β1): {beta1_mult:.3f}")
print(f"Coefficient x2 (β2): {beta2_mult:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].scatter(x_reg, y_reg, alpha=0.7, color='blue', s=50, label='Data')
axes[0,0].plot(x_reg, y_pred, 'r-', linewidth=2, label=f'Regression Line (R² = {r2:.3f})')
axes[0,0].set_title('Simple Linear Regression')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
axes[0,0].legend()

axes[0,1].scatter(y_pred, resid, alpha=0.7, color='green', s=50)
axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0,1].set_title('Residual Plot')
axes[0,1].set_xlabel('Predicted Values')
axes[0,1].set_ylabel('Residuals')

sorted_resid = sorted(resid)
theoretical_quantiles = np.linspace(0, 1, len(resid))
axes[1,0].scatter(theoretical_quantiles, sorted_resid, alpha=0.7, color='purple', s=50)
axes[1,0].plot([0, 1], [min(sorted_resid), max(sorted_resid)], 'r--', alpha=0.5)
axes[1,0].set_title('Q-Q Plot of Residuals')
axes[1,0].set_xlabel('Theoretical Quantiles')
axes[1,0].set_ylabel('Sample Quantiles')

ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
ax_3d.scatter(x1_mult, x2_mult, y_mult, alpha=0.7, s=50, color='red')
ax_3d.set_xlabel('X1')
ax_3d.set_ylabel('X2')
ax_3d.set_zlabel('Y')
ax_3d.set_title('3D Scatter Plot (Multiple Regression)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regression.png'))
plt.show()

# 9. ADVANCED REGRESSION DIAGNOSTICS

def polynomial_regression(x, y, degree=2):
    coeffs = np.polyfit(x, y, degree)
    return coeffs[::-1]

def predict_polynomial(x, coeffs):
    return sum(coeffs[i] * x**i for i in range(len(coeffs)))

# Example polynomial regression
x_poly = np.linspace(1, 10, 100).tolist()
y_poly = [0.5 * x**2 + 1.2 * x + 1 + np.random.normal(0, 2) for x in x_poly]

poly_coeffs = polynomial_regression(x_poly, y_poly, degree=2)
y_poly_pred = [predict_polynomial(xi, poly_coeffs) for xi in x_poly]
r2_poly = r_squared(y_poly, y_poly_pred)

print(f"\nPolynomial Regression (degree 2):")
print(f"Coefficients: {[f'{c:.3f}' for c in poly_coeffs]}")
print(f"R-squared: {r2_poly:.3f}")

# 10. DATA TRANSFORMATIONS

def log_transform(data):
    return [math.log(x) for x in data if x > 0]

def sqrt_transform(data):
    return [math.sqrt(x) for x in data if x >= 0]

def reciprocal_transform(data):
    return [1/x for x in data if x != 0]

def z_score_normalize(data):
    m = mean(data)
    sd = std_deviation(data)
    return [(x - m) / sd for x in data]

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# Example transformations
skewed_data = np.random.exponential(10, 100).tolist()
log_data = log_transform(skewed_data)
sqrt_data = sqrt_transform(skewed_data)
z_data = z_score_normalize(skewed_data)
minmax_data = min_max_normalize(skewed_data)

print(f"\nData Transformations:")
print(f"Original: {skewed_data[:5]}...")
print(f"Log: {[f'{x:.2f}' for x in log_data[:5]]}...")
print(f"Sqrt: {[f'{x:.2f}' for x in sqrt_data[:5]]}...")
print(f"Z-score: {[f'{x:.2f}' for x in z_data[:5]]}...")
print(f"Min-max: {[f'{x:.2f}' for x in minmax_data[:5]]}...")

# 11. COMPREHENSIVE VISUALIZATION

fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(3, 4, 1)
plt.hist(skewed_data, bins=12, alpha=0.7, color='red', edgecolor='black')
plt.title('Original Data (Skewed)')
plt.xlabel('Value')
plt.ylabel('Frequency')

ax2 = plt.subplot(3, 4, 2)
plt.hist(log_data, bins=12, alpha=0.7, color='blue', edgecolor='black')
plt.title('Log Transformed')
plt.xlabel('Log(Value)')
plt.ylabel('Frequency')

ax3 = plt.subplot(3, 4, 3)
plt.scatter(x_poly, y_poly, alpha=0.7, color='green', s=50, label='Data')
x_smooth = np.linspace(1, 10, 100)
y_smooth = [predict_polynomial(xi, poly_coeffs) for xi in x_smooth]
plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Polynomial (R² = {r2_poly:.3f})')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

ax4 = plt.subplot(3, 4, 4)
np.random.seed(42)
x_corr = np.random.normal(0, 1, 100)
y_high_corr = 0.9 * x_corr + 0.1 * np.random.normal(0, 1, 100)
y_low_corr = 0.3 * x_corr + 0.7 * np.random.normal(0, 1, 100)
plt.scatter(x_corr, y_high_corr, alpha=0.7, color='blue', s=30, label='High correlation')
plt.scatter(x_corr, y_low_corr, alpha=0.7, color='red', s=30, label='Low correlation')
plt.title('Correlation Comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

ax5 = plt.subplot(3, 4, 5, projection='3d')
x1_surf = np.linspace(min(x1_mult), max(x1_mult), 20)
x2_surf = np.linspace(min(x2_mult), max(x2_mult), 20)
X1_surf, X2_surf = np.meshgrid(x1_surf, x2_surf)
Y_surf = beta0_mult + beta1_mult * X1_surf + beta2_mult * X2_surf
ax5.plot_surface(X1_surf, X2_surf, Y_surf, alpha=0.7, cmap='viridis')
ax5.scatter(x1_mult, x2_mult, y_mult, color='red', s=50)
ax5.set_xlabel('X1')
ax5.set_ylabel('X2')
ax5.set_zlabel('Y')
ax5.set_title('3D Regression Surface')

ax6 = plt.subplot(3, 4, 6)
x_dist = np.linspace(-4, 4, 100)
y_normal = [normal_pdf(x, 0, 1) for x in x_dist]
y_t = t.pdf(x_dist, 5)
plt.plot(x_dist, y_normal, 'b-', linewidth=2, label='Normal')
plt.plot(x_dist, y_t, 'r--', linewidth=2, label='t-distribution (df=5)')
plt.title('Distribution Comparison')
plt.legend()

ax7 = plt.subplot(3, 4, 7)
chi_x = np.linspace(0, 15, 100)
chi_y = chi2.pdf(chi_x, df)
plt.plot(chi_x, chi_y, 'g-', linewidth=2)
plt.axvline(chi_stat, color='red', linestyle='--', label=f'Test statistic: {chi_stat:.2f}')
plt.title('Chi-square Distribution')
plt.xlabel('Chi-square value')
plt.ylabel('Density')
plt.legend()

ax8 = plt.subplot(3, 4, 8)
x_ci = np.linspace(min(x_reg), max(x_reg), 50)
y_ci = [predict(xi, beta0, beta1) for xi in x_ci]
ci_width = 1.96 * standard_error(y_reg)
y_upper = [y + ci_width for y in y_ci]
y_lower = [y - ci_width for y in y_ci]
plt.plot(x_ci, y_ci, 'b-', linewidth=2, label='Regression')
plt.fill_between(x_ci, y_lower, y_upper, alpha=0.3, color='blue', label='95% CI')
plt.scatter(x_reg, y_reg, alpha=0.7, color='red', s=30)
plt.title('Confidence Intervals')
plt.legend()

ax9 = plt.subplot(3, 4, 9)
plt.scatter(y_pred, resid, alpha=0.7, color='purple', s=50)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')

ax10 = plt.subplot(3, 4, 10)
plt.boxplot([skewed_data, log_data, sqrt_data], labels=['Original', 'Log', 'Sqrt'])
plt.title('Transformation Effects')
plt.ylabel('Value')

ax11 = plt.subplot(3, 4, 11)
corr_data = np.array([
    [1.0, 0.8, -0.6, 0.3],
    [0.8, 1.0, -0.4, 0.2],
    [-0.6, -0.4, 1.0, -0.1],
    [0.3, 0.2, -0.1, 1.0]
])
im = plt.imshow(corr_data, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title('Correlation Matrix')

ax12 = plt.subplot(3, 4, 12)
sample_means = []
for _ in range(100):
    sample = np.random.choice(population, size=50, replace=True)
    sample_means.append(mean(sample))
plt.hist(sample_means, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.title('Sampling Distribution of Mean')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comprehensive_visualization.png'))
plt.show()

# 12. STATISTICAL TESTS SUMMARY

print("\n" + "="*60)
print("STATISTICAL TESTS SUMMARY")
print("="*60)

print("\n1. ONE-SAMPLE T-TEST:")
print("   Tests if sample mean differs from population mean")
print("   H0: μ = μ0")
print("   Use when: Sample size < 30 or population std unknown")

print("\n2. TWO-SAMPLE T-TEST:")
print("   Tests if two sample means differ")
print("   H0: μ1 = μ2")
print("   Use when: Comparing means of two groups")

print("\n3. CHI-SQUARE GOODNESS OF FIT:")
print("   Tests if observed frequencies match expected")
print("   H0: Observed = Expected")
print("   Use when: Categorical data, testing distributions")

print("\n4. CHI-SQUARE INDEPENDENCE:")
print("   Tests if two variables are independent")
print("   H0: Variables are independent")
print("   Use when: Contingency tables, categorical variables")

print("\n5. CORRELATION TEST:")
print("   Tests if correlation coefficient is significant")
print("   H0: ρ = 0")
print("   Use when: Testing linear relationship strength")

# 13. REGRESSION ASSUMPTIONS

print("\n" + "="*60)
print("REGRESSION ASSUMPTIONS")
print("="*60)

print("\n1. LINEARITY:")
print("   - Relationship between X and Y is linear")
print("   - Check: Scatter plot, residual plot")

print("\n2. INDEPENDENCE:")
print("   - Observations are independent")
print("   - Check: Study design, residual patterns")

print("\n3. HOMOSCEDASTICITY:")
print("   - Constant variance of residuals")
print("   - Check: Residual vs fitted plot")

print("\n4. NORMALITY:")
print("   - Residuals are normally distributed")
print("   - Check: Q-Q plot, histogram of residuals")

print("\n5. NO MULTICOLLINEARITY:")
print("   - Independent variables not highly correlated")
print("   - Check: Correlation matrix, VIF")

# 14. EFFECT SIZE MEASURES

def cohens_d(group1, group2):
    mean1, mean2 = mean(group1), mean(group2)
    var1, var2 = variance(group1), variance(group2)
    n1, n2 = len(group1), len(group2)
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (mean1 - mean2) / pooled_std

# Example effect size with increased samples
group_a = np.random.normal(25, 5, 50).tolist()
group_b = np.random.normal(28, 5, 50).tolist()

d = cohens_d(group_a, group_b)
print(f"\nEffect Size (Cohen's d): {d:.3f}")

if abs(d) < 0.2:
    effect_size = "Small"
elif abs(d) < 0.5:
    effect_size = "Medium"
else:
    effect_size = "Large"

print(f"Effect Size Interpretation: {effect_size}")

# 15. POWER ANALYSIS (SIMPLIFIED)

def power_analysis_t_test(effect_size, alpha=0.05, power=0.8):
    if power == 0.8 and alpha == 0.05:
        if effect_size == 0.2:
            return 393
        elif effect_size == 0.5:
            return 64
        elif effect_size == 0.8:
            return 26
    return "Use specialized software for exact calculation"

print(f"\nSample Size for Power = 0.8, α = 0.05:")
print(f"Small effect (d=0.2): {power_analysis_t_test(0.2)} per group")
print(f"Medium effect (d=0.5): {power_analysis_t_test(0.5)} per group")
print(f"Large effect (d=0.8): {power_analysis_t_test(0.8)} per group")
print("Note: Use statistical software for precise power calculations")

# 16. FINAL COMPREHENSIVE EXAMPLE

np.random.seed(123)
n = 200  # Increased sample size
x1_final = np.random.normal(100, 15, n).tolist()
x2_final = np.random.normal(50, 10, n).tolist()
error = np.random.normal(0, 5, n).tolist()
y_final = [10 + 0.5 * x1 + 0.3 * x2 + e for x1, x2, e in zip(x1_final, x2_final, error)]

print(f"\nDataset Summary (n={n}):")
print(f"X1 - Mean: {mean(x1_final):.2f}, SD: {std_deviation(x1_final):.2f}")
print(f"X2 - Mean: {mean(x2_final):.2f}, SD: {std_deviation(x2_final):.2f}")
print(f"Y - Mean: {mean(y_final):.2f}, SD: {std_deviation(y_final):.2f}")

corr_x1_y = correlation(x1_final, y_final)
corr_x2_y = correlation(x2_final, y_final)
corr_x1_x2 = correlation(x1_final, x2_final)

print(f"\nCorrelation Analysis:")
print(f"X1-Y correlation: {corr_x1_y:.3f}")
print(f"X2-Y correlation: {corr_x2_y:.3f}")
print(f"X1-X2 correlation: {corr_x1_x2:.3f}")

beta0_simple, beta1_simple = simple_linear_regression(x1_final, y_final)
y_pred_simple = [predict(xi, beta0_simple, beta1_simple) for xi in x1_final]
r2_simple = r_squared(y_final, y_pred_simple)

print(f"\nSimple Regression (X1 -> Y):")
print(f"Y = {beta0_simple:.3f} + {beta1_simple:.3f}*X1")
print(f"R² = {r2_simple:.3f}")

t_stat_simple, df_simple = t_test_one_sample(y_final, 60)
print(f"\nOne-sample t-test (H0: μ = 60):")
print(f"t-statistic: {t_stat_simple:.3f}, df: {df_simple}")

print(f"\nAnalysis Complete!")
print("="*60)

# Final comprehensive plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

ax1.scatter(x1_final, y_final, alpha=0.6, color='blue', s=30)
ax1.plot([min(x1_final), max(x1_final)], 
         [predict(min(x1_final), beta0_simple, beta1_simple), 
          predict(max(x1_final), beta0_simple, beta1_simple)], 
         'r-', linewidth=2)
ax1.set_xlabel('X1')
ax1.set_ylabel('Y')
ax1.set_title(f'Simple Regression (R² = {r2_simple:.3f})')

residuals_simple = residuals(y_final, y_pred_simple)
ax2.scatter(y_pred_simple, residuals_simple, alpha=0.6, color='green', s=30)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residual Plot')

ax3.hist(residuals_simple, bins=15, alpha=0.7, color='orange', edgecolor='black')
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Residual Distribution')

corr_matrix_final = np.array([
    [1.0, corr_x1_x2, corr_x1_y],
    [corr_x1_x2, 1.0, corr_x2_y],
    [corr_x1_y, corr_x2_y, 1.0]
])
im = ax4.imshow(corr_matrix_final, cmap='RdBu', vmin=-1, vmax=1)
ax4.set_title('Correlation Matrix')
ax4.set_xticks([0, 1, 2])
ax4.set_yticks([0, 1, 2])
ax4.set_xticklabels(['X1', 'X2', 'Y'])
ax4.set_yticklabels(['X1', 'X2', 'Y'])

for i in range(3):
    for j in range(3):
        ax4.text(j, i, f'{corr_matrix_final[i][j]:.2f}', 
                ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_comprehensive.png'))
plt.show()