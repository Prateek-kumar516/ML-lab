import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.rand(20) * 10

Y = 2.5 * X + 5 + np.random.randn(20) * 2

m = np.random.randn()
c = np.random.randn()

learning_rate = 0.01

iterations = 1000

n = len(X)

cost_history = []

for i in range(iterations):

    Y_pred = m * X + c

    error = Y_pred - Y

    cost = np.mean(error ** 2)
    cost_history.append(cost)

    dm = (2 / n) * np.sum(X * error)
    dc = (2 / n) * np.sum(error)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

Y_pred = m * X + c

print("Gradient Descent Linear Regression")
print("----------------------------------")
print("Final slope (m):", m)
print("Final intercept (c):", c)
print(f"Final equation: y = {m:.4f}x + {c:.4f}")
print(f"Final Mean Squared Error: {cost_history[-1]:.4f}")

plt.figure(figsize=(8, 5))

plt.scatter(X, Y, label="Data Points")

order = np.argsort(X)

plt.plot(
    X[order],
    Y_pred[order],
    label="Gradient Descent Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Cost Function during Gradient Descent")
plt.grid(True)
plt.show()