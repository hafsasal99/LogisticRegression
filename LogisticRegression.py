import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data read and split into features and labels
df = pd.read_csv('ex2data1.txt', sep=",", header=None)
x, y = df.iloc[:, :-1], df.iloc[:, [-1]]

# slicing to get not admitted & admitted data points
X0 = df[df.iloc[:,2] == 0]
X1 = df[df.iloc[:,2] == 1]

# visualizing data
plt.scatter(X0.iloc[:, 0], X0.iloc[:, 1], marker='o', color="yellow", label='Not Admitted')
plt.scatter(X1.iloc[:, 0], X1.iloc[:, 1], marker='+', color="black", label='Admitted')
plt.title('Figure 1: Scatter Plot of Training Data')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()

# normalizing data
means = pd.DataFrame(x).mean()
sigmas = pd.DataFrame(x).std()
x = (x-means)/sigmas
x.insert(0, 'dummy', 1)

# weights
weights = np.random.rand((len(x.columns)))
print(weights)

# predicted value
sample_Size = len(x.index)
x = x.transpose()
predicted = np.dot(weights.transpose(), x)
predicted = pd.Series(predicted)

# sigmoid
predicted = 1/(1 + np.exp(-predicted))
y = y.squeeze()
predicted = predicted.squeeze()
print(predicted)

# cost function
#cost = -(1/sample_Size) * np.sum((y * np.log(predicted)) + ((1-y) * np.log(1-predicted))if ( predicted != 1 and predicted != 0 ) else 0) * 1/sample_Size
cost = sum((-y * np.log(predicted)) - ((1-y) * np.log(1-predicted))) / sample_Size
print(cost)


# gradient descent
count = 0
trainingRate = 0.05
costlist = []
while cost > 0 and count < 1000 :

    # weight update rule
    weights = pd.Series(weights)
    featureVector = x.transpose()
    new_weights = weights.subtract(trainingRate * sum(np.dot((predicted.subtract(y)).to_frame().transpose(),featureVector)).transpose() * 1 / len(x.index))
    weights = new_weights
    print(weights)

    # calculating h0
    predicted = np.dot(weights.transpose(), x)
    predicted = pd.Series(predicted)
    predicted = 1 / (1 + np.exp(-predicted))
    predicted = pd.Series(predicted)
    y = y.squeeze()
    predicted = predicted.squeeze()


    cost = sum((-y * np.log(predicted)) - ((1 - y) * np.log(1 - predicted))) / sample_Size
    if count % 5 == 0:
        costlist.append(cost)
    count += 1
    print(cost)

# Convergence Curve
costlist = np.asarray(costlist)
count=list(range(0, count, 5))
count=np.asarray(count)
plt.plot(count, costlist, '-r')  # plot the cost function.
plt.show()

# # drawing decision boundary
X0 = df[df.iloc[:,2] == 0]
X1 = df[df.iloc[:,2] == 1]
X0,Y0 = X0.iloc[:,0],X0.iloc[:,1]
X1,Y1 = X1.iloc[:,0],X1.iloc[:,1]
X0 = (X0 - means[0])/sigmas[0]
Y0 = (Y0 -means[1])/sigmas[1]
X1 = (X1 - means[0])/sigmas[0]
Y1 = (Y1 - means[1])/sigmas[1]
plt.scatter(X0,Y0, marker='o', color="yellow", label='Not Admitted')
plt.scatter(X1 ,Y1, marker='+', color="black", label='Admitted')
x = x.transpose()
x1 = [np.max(x.iloc[:,1]),np.min(x.iloc[:,1])]
weights=weights.transpose()
y_values = - (weights[0] + np.dot(weights[1], x1)) / weights[2]
plt.plot(x1, y_values, label='Decision Boundary')
plt.title('Figure 1: Scatter Plot of Training Data')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()


# testing data
test_features=[1,45.0,85.0]
test_features[1] = (test_features[1]-means[0])/sigmas[0]
test_features[2] = (test_features[2]-means[1])/sigmas[1]
test_features=np.array(test_features)
test_features = test_features.transpose()
predicted = np.dot(weights.transpose(), test_features)
predicted = 1/(1 + np.exp(-predicted))
print('The probability of this student getting admission is  ', predicted)