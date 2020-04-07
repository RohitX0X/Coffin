import numpy as np
def htheta(W,X):
    # print("hell1")
    z=X.dot(W)
    #print(z.shape,"    ",X.shape," ",W.shape)
    z=np.float32(z)
    return 1.0/(1+ np.exp(-z))

def cost(W,X,y):
    m=X.shape[0]
    z=htheta(W,X)
    # print("hellr")
    costu=sum(((y)*np.log(z)) + (1-y)*np.log(1-z))/m
    # print(costu)
    return -1*costu

def grad_descent(W,X,y,alpha):
    m=X.shape[0]

    gradient = np.dot(X.T,(htheta(W,X) - y))/m
    
   # print("hell2"," ",X.T.shape," ",y.shape," ",htheta(W,X).shape," ",gradient.shape ) 
    W=W-alpha*gradient
    
    #print(W.shape)
    return W
def train(X, y, W, alpha, iters):
    cost_history = []
    m=X.shape[0]
    for i in range(iters):
        print(i)
        W= grad_descent(W,X,y,alpha)
        costu = cost(W, X, y)
        cost_history.append(costu)
        print(costu)
        # Log Progress
        if i % 100 == 0:
            print ("iter: "+str(i) + " cost: "+str(cost))

    return W, cost_history


