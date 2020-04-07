function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m)
    return cost
end

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = (X' * loss)/m
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X, Y, B)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B, costHistory
end

function RMSS(X,Y,newB)
    m=length(Y)
    rmss=sqrt(sum((X*newB-Y).^2)/m)
    return rmss
end

function R2(X,Y,newB,Ymean)
    m=length(Y)
    r2=1.0 - (sum((Y-X*newB).^2)/sum((Y-Ymean*ones(m)).^2))
    # print(sum((Y-X*newB).^2))
    return r2
end

function RidgeCost(X,Y,B,alph)
    m = length(Y)
    ### I am dividing the entire cost by 2m so as to scale it down
    ### as we are using large values and it doesnt effect the weights
    ### as scaling doesnt change the minimization probem
    cost = sum(((X * B) - Y).^2)/(2*m) +  alph*sum(B[2:end].^2)/(2*m)
    return cost
end

function gradientDescentRidge(X,Y,B,alph,learningRate,numIter)
    costHistory = zeros(numIter)
    m = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIter
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle

        gradient = (X' * loss)/m + alph*B/m
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = RidgeCost(X, Y, B,alph)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B,costHistory
end

function LassoCost(X,Y,B,alph)
    m = length(Y)
    ### I am dividing the entire cost by 2m so as to scale it down
    ### as we are using large values and it doesnt effect the weights
    ### as scaling doesnt change the minimization probem
    cost = sum(((X * B) - Y).^2)/(2*m) + alph*sum(abs.(B[2:end]))/(2*m)
    return cost
end

function gradingLasso(X,Y,B,alph,learningRate,numIter )
    costHistory = zeros(numIter)
    m = length(Y)
    n=size(X)[2]
    # X = normalized(X,m) #normalizing X in case it was not done before
    for i in 1:numIter
       for j in 1:n
           X_j = X[:,j]
           l2_X_j = sum(X_j.^2)/m
           y_pred = X*B
           rho = X_j'*(Y - y_pred  + B[j]*X_j)/m
           rho=sum(rho)
           B[j] =  soft_threshold(rho, alph)/l2_X_j
       end
       cost = LassoCost(X, Y, B,alph)
       # Store costs in a vairable to visualize later
       costHistory[i] = cost
   end
   return B,costHistory
end

function normalized(X,m)
    A=zeros(size(X)[1],size(X)[2])
    for i in 1:size(X)[2]
        A[:,i]=normalize(X[:,i])
    end
    return A
end

function soft_threshold(rho,alph)
    if rho < - alph
        return (rho + alph)
    elseif rho >  alph
        return (rho - alph)
    else
        return 0
    end
end
