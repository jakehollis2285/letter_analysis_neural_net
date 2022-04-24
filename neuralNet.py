import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_parameters(in_dim, out_dim):
    w = np.random.randn(in_dim, out_dim)*0.01
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[0]

    unique, counts = np.unique(Y, return_counts=True)
    result = np.column_stack((unique, counts)) 
    
    #calculate activation function
    A = sigmoid(np.dot(w.T, X.T)+b).T

    ret = []

    for i in range(A.shape[0]):
        mmax = 0
        jmax = None
        for j in range(A.shape[1]):

            if(mmax == 0 or A[i][j] > mmax):
                mmax = A[i][j]
                jmax = j

        ret.append(jmax)

    A = np.array([ret])

    Ap = np.zeros((m, len(result)))
    Yp = np.zeros((m, len(result)))

    for i in range(len(A[0])):
        for j in range(len(Ap[1])):
            if(j == A[0][i]):
                Ap[i][j] = A[0][i]

    for i in range(len(Y[0])):
        for j in range(len(Yp[1])):
            if(j == Y[0][i]):
                Yp[i][j] = Y[0][i]

    A = Ap / 26
    Y = Yp / 26
    A[A == 0] = np.finfo(float).eps
    Y[Y == 0] = np.finfo(float).eps

    #find the cost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  
    #find gradient (back propagation)
    dw = (1/m) * np.dot(X.T, (A-Y))
    db = (1/m) * np.sum(A-Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db} 
    return grads, cost


def gradient_descent(w, b, X, Y, iterations, learning_rate):
    costs = []
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        #update parameters
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        costs.append(cost)
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}    
    return params, costs

def predict(w, b, X, out_dim):
    # number of example
    m = X.shape[1]
    y_pred = np.zeros((m, out_dim))
    w = w.reshape(m, out_dim)

    A = sigmoid(np.dot(w.T, X.T)+b).T
    
    y_pred[np.arange(len(A)), A.argmax(1)] = 1
    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    ret = []

    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            if(y_pred[i][j] == 1):
                ret.append(j)

    return ret


def model(train_x, train_y, test_x, test_y, iterations, learning_rate):
    unique, counts = np.unique(train_y, return_counts=True)
    result = np.column_stack((unique, counts)) 

    w, b = initialize_parameters(train_x.shape[1], len(result))
    parameters, costs = gradient_descent(w, b, train_x, train_y, iterations, learning_rate)
    
    w = parameters["w"]
    b = parameters["b"]
    
    # predict 
    train_pred_y = predict(w, b, train_x, len(result))
    unique, counts = np.unique(test_y, return_counts=True)
    result = np.column_stack((unique, counts)) 
    test_pred_y = predict(w, b, test_x, len(result))
    print("Train Acc: {} %".format(100 - np.mean(np.abs(train_pred_y - train_y)) * 100))
    print("Test Acc: {} %".format(100 - np.mean(np.abs(test_pred_y - test_y)) * 100))
    
    return train_pred_y, test_pred_y, costs