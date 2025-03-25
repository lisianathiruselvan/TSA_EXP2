# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
NAME:LISIANA T
REG NO:212222240053
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/content/World Population.csv')

# Extract Year from "Date" column (taking first two characters and converting to full year)
data['Year'] = data['Date'].astype(str).str[:2].astype(int) + 2000  # Assuming years are 2000+

# Remove "%" from the "Percentage" column and convert it to a numeric value
data['Percentage'] = data['Percentage'].str.replace('%', '').astype(float)

# Group by Year and compute the mean percentage (assuming world total percentage remains ~100%)
resampled_data = data.groupby('Year', as_index=False)['Percentage'].mean()

# Extract values
years = resampled_data['Year'].tolist()
percentage = resampled_data['Percentage'].tolist()

# Preprocessing for Trend Calculation
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, percentage)]
# Linear Trend Estimation
n = len(years)
b = (n * sum(xy) - sum(percentage) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(percentage) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]
# Polynomial Trend Estimation (Degree 2)
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, percentage)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(percentage), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]
# Print trend equations
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Add trends to dataset
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

# Set index to 'Year'
resampled_data.set_index('Year', inplace=True)

# Visualization
resampled_data['Percentage'].plot(kind='line', color='blue', marker='o', label='Actual Percentage')
resampled_data['Linear Trend'].plot(kind='line', color='black', linestyle='--', label='Linear Trend')
resampled_data['Polynomial Trend'].plot(kind='line', color='red', marker='o', label='Polynomial Trend')

plt.xlabel('Year')
plt.ylabel('Population Percentage (%)')
plt.legend()
plt.title('World Population Percentage Trend Estimation')
plt.grid()
plt.show()+
```

### OUTPUT
![image](https://github.com/user-attachments/assets/b9477cea-33ba-4ba2-a485-842008ffb338)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
