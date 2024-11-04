import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('housing_retail_price.csv.csv')

# Extract the feature (SquareFootage) and the target (Price)
X = data['SquareFootage'].values
y = data['Price'].values

# Add a column of ones to X for the intercept term (bias)
X_b = np.c_[np.ones((len(X), 1)), X]  # X_b becomes a 2D array with an additional column of 1s

# Calculate the optimal theta (coefficients) using the Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Extract intercept and slope from theta_best
intercept = theta_best[0]
slope = theta_best[1]

# Make predictions for the test data (X)
y_pred = intercept + slope * X

# Print the results
print(f"Intercept (Theta_0): {intercept}")
print(f"Slope (Theta_1): {slope}")

# Plot the data and the linear regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Linear Regression ')
plt.legend()
plt.show()
