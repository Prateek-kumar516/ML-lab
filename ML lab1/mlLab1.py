data = list(map(float, input("Enter numbers separated by space: ").split()))

n = len(data)

mean = sum(data) / n

sorted_data = sorted(data)
if n % 2 == 0:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
else:
    median = sorted_data[n // 2]

freq = {}
for num in data:
    if num in freq:
        freq[num] += 1
    else:
        freq[num] = 1

max_freq = max(freq.values())
modes = []

for key in freq:
    if freq[key] == max_freq:
        modes.append(key)

if max_freq == 1:
    mode = "No Mode"
else:
    mode = modes

data_range = max(data) - min(data)

variance = 0
for num in data:
    variance += (num - mean) ** 2
variance /= n

std_deviation = variance ** 0.5

print("\n----- Results -----")
print("Mean =", mean)
print("Median =", median)
print("Mode =", mode)
print("Range =", data_range)
print("Variance =", variance)
print("Standard Deviation =", std_deviation)