#  Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
import numpy as np
import matplotlib.pyplot as plt
import mnist
from statistics import mean 

def sqDistance(p, q, pSOS, qSOS):
    #  Efficiently compute squared euclidean distances between sets of vectors
    #  Compute the squared Euclidean distances between every d-dimensional point
    #  in p to every d-dimensional point in q. Both p and q are
    #  npoints-by-ndimensions. 
    #  d(i, j) = sum((p(i, :) - q(j, :)).^2)
    d = np.add(pSOS, qSOS.T) - 2*np.dot(p, q.T)
    return d

np.random.seed(10)

#  Set training & testing and randomizing.

Xtrain, ytrain, Xtest, ytest = mnist.load_data()
all_idx = np.array(np.arange(len(Xtrain)))
all_idx=np.random.choice(all_idx,size=60000)

Xtrain = [Xtrain[ii] for ii in all_idx]
Xtrain=np.array(Xtrain)
ytrain = [ytrain[ii] for ii in all_idx]
ytrain=np.array(ytrain)

def knn(train_size,Xtrain, ytrain, Xtest, ytest):
    #train_size = 10000
    test_size  = 10000
    
    Xtrain = Xtrain[0:train_size]
    ytrain = ytrain[0:train_size]
    
    Xtest = Xtest[0:test_size]
    ytest = ytest[0:test_size]
    
    #  Precompute sum of squares term for speed
    XtrainSOS = np.sum(Xtrain**2, axis=1, keepdims=True)  #axis=1 means along the row add -->
    XtestSOS  = np.sum(Xtest**2, axis=1, keepdims=True)
    
    #  fully solution takes too much memory so we will classify in batches
    #  nbatches must be an even divisor of test_size, increase if you run out of memory 
    if test_size > 1000:
      nbatches = 50
    else:
      nbatches = 5
    batches = np.array_split(np.arange(test_size), nbatches)

    ypred = np.zeros_like(ytest)
    
    #  Classify
    for i in range(nbatches):
        dst = sqDistance(Xtest[batches[i]], Xtrain, XtestSOS[batches[i]], XtrainSOS)
        closest = np.argmin(dst, axis=1)
        ypred[batches[i]] = ytrain[closest]
    
    #  Report
    errorRate = (ypred != ytest).mean()
    print('Error Rate: {:.2f}%\n'.format(100*errorRate))
    
    #  image plot
    plt.imshow(Xtrain[0].reshape(28, 28), cmap='gray')
    plt.show()
    
    return errorRate

def val_knn(val,Xtrain, ytrain, Xtest, ytest):
    train_size = 1000
    Xtrain = Xtrain[0:train_size]
    ytrain = ytrain[0:train_size]
    errorRate=[0]*val
    if train_size%val == 0:
            test_size  = train_size//val
    else:
            test_size = train_size//val
            
    for i in range(val):
        new_Xtrain = np.delete(Xtrain,slice((i*test_size),((i+1)*test_size)),axis=0) #Removes the validation set from the train set
        new_ytrain = np.delete(ytrain,slice((i*test_size),((i+1)*test_size)),axis=0) #Removes the validation set from the train labels
        new_Xtest = Xtrain[(i*test_size):((i+1)*test_size)] #Create a new validation set
        new_ytest = ytrain[(i*test_size):((i+1)*test_size)] #Create validation set labels
        
        #  Precompute sum of squares term for speed
        XtrainSOS = np.sum(new_Xtrain**2, axis=1, keepdims=True)  #axis=1 means along the row add -->
        XtestSOS  = np.sum(new_Xtest**2, axis=1, keepdims=True)
        
        #  full solution takes too much memory so we will classify in batches
        if test_size > 1000:
          nbatches = 50
        elif test_size==333:
            nbatches=3
        else:
          nbatches = 5

        batches = np.array_split(np.arange(test_size), nbatches)
        ypred = np.zeros_like(new_ytest)
        
        #  Classify
        for ii in range(nbatches):
            dst = sqDistance(new_Xtest[batches[ii]], new_Xtrain, XtestSOS[batches[ii]], XtrainSOS)
            closest = np.argmin(dst, axis=1)
            ypred[batches[ii]] = new_ytrain[closest]
        
        #  Report
        errorRate[i] = (ypred != new_ytest).mean()
        print('Error Rate: {:.2f}%\n'.format(100*errorRate[i]))
        print("loading")
    print("done")
    return errorRate

    # Q1:  Plot a figure where the x-asix is number of training
    #      examples (e.g. 100, 1000, 2500, 5000, 7500, 10000), and the y-axis is test error.
    
    # TODO

errors=[]
errors.append(knn(10000,Xtrain, ytrain, Xtest, ytest))
errors.append(knn(5000,Xtrain, ytrain, Xtest, ytest))
errors.append(knn(2500,Xtrain, ytrain, Xtest, ytest))
errors.append(knn(1000,Xtrain, ytrain, Xtest, ytest))
errors.append(knn(200,Xtrain, ytrain, Xtest, ytest))
    
num_training=[10000,5000,2500,1000,200]



# Q2:  plot the n-fold cross validation error for the first 1000 training training examples

# TODO
e1=val_knn(3,Xtrain, ytrain, Xtest, ytest)
e2=val_knn(10,Xtrain, ytrain, Xtest, ytest)
e3=val_knn(50,Xtrain, ytrain, Xtest, ytest)
e4=val_knn(100,Xtrain, ytrain, Xtest, ytest)
e5=val_knn(1000,Xtrain, ytrain, Xtest, ytest)
print("")
e1=mean(e1)
e2=mean(e2)
e3=mean(e3)
e4=mean(e4)
e5=mean(e5)

n_values=[3,10,50,100,1000]

####plot for Q1
print("Plot for Q1")
plt.figure()
plt.xlabel('Training examples')
plt.ylabel('Error')
plt.plot(num_training,errors, label='Error')
plt.legend()
plt.show()

####Plot for Q2
print("Plot for Q2")
print("Errors",e1,e2,e3,e4,e5)
plt.figure()
plt.xlabel('value of n')
plt.ylabel('Cross validation error')
plt.plot(n_values,[e1,e2,e3,e4,e5], label='Error')
plt.legend()
plt.show()