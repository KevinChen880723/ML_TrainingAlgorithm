# ML_TrainingAlgorithm

## Description
In this project, I implement two training algorithms called Stochastic Gradient Descent and Linear Regression.

### Stochastic Gradient Descent (SGD)
I use two different kind of error measurement: 
1. Cross Entropy
This is the Logistic Regression algorithm optimized by SGD, so predict the result by following equation: y = θ(W^T●X) then you will get the result in the range of [-1, +1].
```python=
def CrossEntropy(y_hat, y):
    return np.sum(-np.log(Sigmoid(y_hat * y)))/len(y)
```
2. Squared error
This is actually Linear Regression. The different place with LinearRegression function is get the result by SGD but not the closed form solution.
```python=
def SquaredError(y_hat, y):
    return (np.linalg.norm(y_hat-y))**2/len(y)
```

### Linear Regression
The ouput value of this algorithm will approximate to the ground truth. It's value is not limited in the range of [-1, +1].
The upper bound of Eout - Ein of this algorithm is not that tight. But it is good to be the w0 of other training algorithms like SGD.
```python=
def LinearRegression(datas, labels):
    pseudoInverse = np.linalg.pinv(datas)
    weight = np.dot(pseudoInverse,labels)
    y = np.dot(datas, weight)
    return y, weight
```

### Feature Transformation
The function is used to Transfer input datas to non-linear hyperplane. So it may help to solve complex problem.
```python=
def FeatureTrasform(datas, Q):
    n_datas = []
    for d in datas:
        n_data = []
        for i in range(Q+1):
            if i == 0:
                n_data.append(1)
            else:
                for j in range(len(d)-1):
                    n_data.append(d[j+1]**(i))
        n_datas.append(n_data)
    return np.array(n_datas)
```
