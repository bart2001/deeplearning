import numpy as np

# load data from csv
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# print
print(x_data.shape, x_data, len(x_data))
print('-------------------------')
print(y_data.shape, y_data)


