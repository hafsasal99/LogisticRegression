#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[3]:


df = pd.read_csv('ex2data2.txt', sep=",", header=None)
x, y = df.iloc[:, :2], df.iloc[:, 2]

# In[4]:


# Binary classification into two groups (Accepted or Rejected)
G0 = df[df.iloc[:, 2] == 0]
G1 = df[df.iloc[:, 2] == 1]
sample_Size = len(x.index)
# Visualizing data
plt.scatter(G0.iloc[:, 0], G0.iloc[:, 1], marker='o', color='yellow', label='y = 0')
plt.scatter(G1.iloc[:, 0], G1.iloc[:, 1], marker='+', color='black', label='y = 1')
plt.xlabel("Microship Test 1")
plt.ylabel("Microchip Test 2")
plt.title("Figure 2: Scatter Plot of Training Data")
plt.legend()
plt.show()

n1 = df.iloc[:, 1]
n2 = df.iloc[:, 2]

# Normalizing data
means = x.mean()
sigmas = x.std()
print(means)
print(sigmas)
#x = (x - means) / sigmas
x.insert(0, 'dummy', 1)
print(x)

# In[5]:


feat1 = x.iloc[:, 1]
feat2 = x.iloc[:, 2]

# Feature Mapping
degree = 6
fmap = np.ones(x.shape[0])[:, np.newaxis]
for i in range(1, degree + 1):
    for j in range(i + 1):
        fmap = np.hstack((fmap, np.multiply(np.power(feat1, i - j), np.power(feat2, j))[:, np.newaxis]))
        x = pd.DataFrame(fmap)
print(x)

# In[6]:


# weights
iweight = np.zeros(x.columns.shape[0])
print(len(x))
# predicted value
x = x.transpose()
predicted = np.dot(iweight.transpose(), x)
predicted = pd.Series(predicted)

# Sigmoid Function
predicted = 1 / (1 + np.exp(-predicted))
y = y.squeeze()
predicted = predicted.squeeze()
print(predicted)


# In[7]:


# Regularized Cost Function
def regCostFunc(lmbda, predicted, weight):
    penalize = (lmbda / (2 * sample_Size)) * sum(np.power(weight, 2))
    cost = (sum((-y * np.log(predicted)) - ((1 - y) * np.log(1 - predicted))) + penalize) / sample_Size
    return cost


# In[8]:


# Initial Cost with Inital Weights

cost2 = regCostFunc(1, predicted, iweight)
print(f'With initial values of theta = 0, the Cost is {cost2}')


# In[9]:


# Gradient Descent
def gradDescent(lmbda, iweight, predicted, y):
    count = 0
    costlist = []
    trainingRate = 0.1

    while cost2 > 0 and count < 40:

        # Weight update rule
        iweight = pd.Series(iweight)
        featureVector = x.transpose()

        # Parameter 0
        new_weights = iweight.subtract(trainingRate * sum(
            np.dot((predicted.subtract(y)).to_frame().transpose(), featureVector)).transpose() * 1 / len(x.index))
        # Added penalty to parameters > 0
        new_weights[1:] = new_weights[1:] + ((lmbda / sample_Size) * sum(np.power(iweight, 2)))
        iweight = new_weights

        # Hypothesis
        predicted = np.dot(iweight.transpose(), x)
        predicted = pd.Series(predicted)
        predicted = 1 / (1 + np.exp(-predicted))
        predicted = pd.Series(predicted)
        y = y.squeeze()
        predicted = predicted.squeeze()

        cost = regCostFunc(1, predicted, iweight)
        if count % 5 == 0:
            costlist.append(cost)
        count += 1
        print(cost)
    return costlist, count, iweight


costlist, count, optweight = gradDescent(1, iweight, predicted, y)

# In[10]:


# Convergence Curve
costlist = np.asarray(costlist)
print(costlist)
count = list(range(0, count, 5))
count = np.asarray(count)
plt.plot(count, costlist, '-r')  # plot the cost function.
plt.show()

# Decision boundary beyond data points
feat1 = x.iloc[:, 1]
feat2 = x.iloc[:, 2]

def mapFeaturePlot(x1, x2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            terms = (x1 ** (i - j) * x2 ** j)
            out = np.hstack((out, terms))
    return out


mask = y == 1
X = df.iloc[:, :-1]
passed = plt.scatter(X[mask][0], X[mask][1], marker='o', color='yellow', label='y = 0')
failed = plt.scatter(X[~mask][0], X[~mask][1], marker='+', color='black', label='y = 1')
# Plotting decision boundary

u_vals = np.linspace(-1, 1.5, 50)
v_vals = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u_vals), len(v_vals)))
for i in range(len(u_vals)):
    for j in range(len(v_vals)):
        z[i, j] = mapFeaturePlot(u_vals[i], v_vals[j]) @ optweight

plt.contour(u_vals, v_vals, z.T, 0)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)
plt.show()

# In[ ]:




