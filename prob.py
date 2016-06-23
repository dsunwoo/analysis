import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import collections

# Create dataset of 1000 random integers
test_data = np.random.random_integers(1, 10, size=1000)
data = test_data.tolist()
c = collections.Counter(test_data)
count_sum = sum(c.values())
print(data)

for k, v in c.items():
    print("The frequency of number {0} is {1}."
          .format(str(k), str(float(v) / count_sum)))

# Create Box Plot
fig = plt.figure()
plt.boxplot(data)
fig.suptitle('BoxPlot: Random # 1-10', fontsize=20)
plt.savefig("data_boxplot.png")
# Create Histogram
fig = plt.figure()
plt.hist(data, histtype='bar')
fig.suptitle('Histogram: Random # 1-10', fontsize=20)
plt.xlabel('Numbers (integer)')
plt.xticks(range(1, 11))
plt.savefig("data_histogram.png")
# Create QQ-Plots
fig = plt.figure()
graph1 = stats.probplot(data, dist="norm", plot=plt)
fig.suptitle('Probability Plot: Random # 1-10', fontsize=20)
plt.savefig("data_probplot.png")
