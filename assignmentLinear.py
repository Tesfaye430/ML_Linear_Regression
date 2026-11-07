import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Sample Dataset: Hours Studied vs. Exam Score
# X = Independent variable (Hours Studied)
# Y = Dependent variable (Exam Score out of 100)
hours = np.array(
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
).reshape(-1, 1)
scores = np.array([35, 40, 48, 53, 58, 62, 67, 75, 82, 88, 94, 98])

# 2. Initialize and Train the Linear Regression Model
model = LinearRegression()
model.fit(hours, scores)

# 3. Extract model parameters (Equation: Y = aX + b)
slope = model.coef_[0]  # Coefficient 'a' (slope)
intercept = model.intercept_  # Intercept 'b'
print(
    f'Trained Model Equation: Score = {slope:.2f} * Hours + {intercept:.2f}\n'
)

# 4. Predict score for a new student (e.g., studying 5.5 hours)
new_hours = np.array([[5.5]])
predicted_score = model.predict(new_hours)
print(
    f'Predicted score for {new_hours[0][0]} hours studied:'
    f' {predicted_score[0]:.2f}\n'
)

# 5. Plotting the Data and the Best-Fit Regression Line
plt.figure(figsize=(9, 5))
plt.scatter(hours, scores, color='blue', label='Actual Student Data', s=70)
plt.plot(
    hours,
    model.predict(hours),
    color='red',
    linewidth=2.5,
    label=f'Best-Fit Line: $Y = {slope:.2f}X + {intercept:.2f}$',
)

# Formatting the plot
plt.title(
    'Linear Regression: Exam Scores vs. Hours Studied',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Score (Out of 100)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()

