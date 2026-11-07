"""A small, step-by-step logistic regression example."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


# Step 1: Make random input values. The seed makes the example repeatable.
np.random.seed(42)
# Generates 20 random input values between -5 and 5. These are the model’s input data.
x = np.linspace(-5, 5, 10)


# Step 2: Make binary labels (0 or 1) with a sigmoid-shaped probability.
# These values are only used to create sample data; the model must learn them.
b = -0.5
w = 1.2
probability = 1 / (1 + np.exp(-(b + w * x)))
# the first number 1 means one trial for each probability (0 or 1).
# The second number, probability, is the chance of success on each trial.
y = np.random.binomial(1, probability)

# Step 3: Give the input to scikit-learn as a table: 50 rows, 1 feature.
X = x.reshape(-1, 1)

# Step 4: Train the logistic regression model on the inputs and labels.
model = LogisticRegression()
model.fit(X, y)

# Step 5: Ask the trained model for probabilities along a smooth range of x.
x_line = np.linspace(-5, 5, 50).reshape(-1, 1)
y_line = model.predict_proba(x_line)[:, 1]

# Step 6: Plot the generated labels and the learned sigmoid curve.
plt.scatter(x, y, color="red", label="Generated data (0 or 1)")
plt.plot(x_line, y_line, color="blue", label="Learned sigmoid")
plt.axhline(0.5, color="gray", linestyle="--", label="0.5 probability")
plt.xlabel("Input x")
plt.ylabel("Probability of class 1")
plt.title("Logistic Regression")
plt.ylim(-0.1, 1.1)
plt.grid(True)
# plt.legend()
plt.show()
