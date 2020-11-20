import os
import numpy as np

def ReadFile(fileName):
    f = open(fileName, mode='r')
    return f.readlines()

def GetTrainingData(rawData):
    '''
    Return data include:
    training_data: 
        A 2-D numpy array which first dimension is training datas; 
        second dimension are the features of each data.
    training_label:
        A 1-D numpy array which store all labels. Index 0 infer the label of first data.
    '''
    data_list_str = []
    
    for data in rawData:
        data_list_str.append(data.split())
    A = np.array(data_list_str)
    A = A.astype('float32')

    training_data = []
    training_label = []
    x0 = []
    for a in A:
        training_data.append(a[0:10])
        training_label.append(a[10])
        x0.append([1.0])
    x0 = np.array(x0)
    training_data = np.array(training_data)
    training_label = np.array(training_label)
    # add x0 = 1 to every xn
    training_data = np.concatenate([x0, training_data], axis=1)
    return (training_data, training_label)

def SquaredError(y_hat, y):
    return (np.linalg.norm(y_hat-y))**2/len(y)

def CrossEntropy(y_hat, y):
    return np.sum(-np.log(Sigmoid(y_hat * y)))/len(y)

def ZeroOneErr(y_hat, y):
    Bool = (np.sign(y_hat) != y)
    n = np.sum(Bool)
    return n/len(y)


def Sigmoid(x):
    return 1/(1+np.exp(-x))

# Use the Linear Regression algorithm to predict the output that approximate to y.
# Output values are not limited in the range of [-1, +1]. They can be any value that approximate to y.
def LinearRegression(datas, labels):
    pseudoInverse = np.linalg.pinv(datas)
    weight = np.dot(pseudoInverse,labels)
    y = np.dot(datas, weight)
    return y, weight

# The SGD algorithm that evaluate error by using squared error.
def SGD_SQR(datas, labels, ita, w0, Err_lin):
    
    w = w0
    update_t = 0
    datas = np.array(datas)
    labels = np.array(labels)
    while SquaredError(np.dot(datas, w), labels) > 1.01*Err_lin:
        i = np.random.randint(0, len(labels))
        w = w+ita*2*np.dot(labels[i]-np.dot(w, datas[i]), datas[i])
        update_t += 1
    return update_t

# The SGD algorithm that evaluate error by using cross entropy.
def SGD_CE(datas, labels, ita, w0, Err_lin):
    
    w = w0
    for repeat in range(500):
        i = np.random.randint(0, len(labels))
        p = -labels[i]*np.dot(w, datas[i])
        w = w + (ita * Sigmoid(p) * labels[i] * datas[i])
    return CrossEntropy(np.dot(datas, w), labels)

# Transfer input datas to non-linear hyperplane
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

def start():
    # Get training datas and testing datas.
    rawData = ReadFile('hw3_train.dat')
    (datas, labels) = GetTrainingData(rawData)
    rawData2 = ReadFile('hw3_test.dat')
    (datas_test, labels_test) = GetTrainingData(rawData2)

    # Do the linear transformation to Q-th Polynomial hypothesis.
    datasT3 = FeatureTrasform(datas, 3)
    datas_testT3 = FeatureTrasform(datas_test, 3)

    (y_lin, W_lin) = LinearRegression(datasT3, labels)

    # Calculate the error between Ein and Eout by using 0/1 error
    print(ZeroOneErr(np.dot(datas_testT3, W_lin), labels_test) - ZeroOneErr(y_lin, labels))

if __name__ == "__main__":
    start()