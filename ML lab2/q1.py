import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

n = len(x)

m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
    (n * np.sum(x ** 2) - (np.sum(x)) ** 2)

c = (np.sum(y) - m * np.sum(x)) / n

y_pred = m * x + c

print("Least Square Approximation\n")
print("Slope (m):", m)
print("Intercept (c):", c)
print("Equation of best-fit line:")
print(f"y = {m:.4f}x + {c:.4f}")

print("\nPredicted values:")
for i in range(n):
    print(f"x = {x[i]}, Actual y = {y[i]}, Predicted y = {y_pred[i]:.4f}")

plt.scatter(x, y, label="Data Points")

plt.plot(x, y_pred, label="Least Square Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Least Square Approximation")
plt.legend()
plt.grid(True)
plt.show()