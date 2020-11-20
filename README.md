# ML_TrainingAlgorithm

## Description
In this project, I implement two training algorithms called Stochastic Gradient Descent and Linear Regression.

### Stochastic Gradient Descent
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
