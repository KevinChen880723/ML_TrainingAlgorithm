# ML_TrainingAlgorithm

## Description
In this project, I implement two training algorithms called Stochastic Gradient Descent and Linear Regression.

### Stochastic Gradient Descent (SGD)
I use two different kind of error measurement: 
1. Cross Entropy
```python=
def CrossEntropy(y_hat, y):
    return np.sum(-np.log(Sigmoid(y_hat * y)))/len(y)
```
2. Squared error
```python=
def SquaredError(y_hat, y):
    return (np.linalg.norm(y_hat-y))**2/len(y)
```

### Linear Regression
The ouput value of this algorithm will approximate to the ground truth. It's value is not limited in the range of [-1, +1].
The upper bound of Eout - Ein of this algorithm is not that tight. But it is good to be the w0 of other training algorithms like SGD.

### Feature Transformation
The function is used to Transfer input datas to non-linear hyperplane. So it may help to solve complex problem.
