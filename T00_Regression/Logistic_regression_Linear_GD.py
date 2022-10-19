# Author: Ahmed Alkhayal - EE482
import numpy as np
import matplotlib.pyplot as plt

# ------------------ generating data ------------------ #

# set seed
np.random.seed(123)

# Create main tensor: 3D matrix for RGB
data_2d = np.zeros((1000,1000))         # [1000x1000]

# generate data of 2 different class at random indices
idx_class0 = np.round(np.random.rand(125,2)*150+700)            # random indices between 700 and 850
idx_class1 = np.array([np.round(np.random.rand(75,2)*250+50),
                       np.round(np.random.rand(75,2)*350+[500,50]),
                       np.round(np.random.rand(75,2)*350+[50,500])])


# ------------------ plotting data ------------------ #

plt.figure(figsize=(5,5))
# background
plt.imshow(data_2d, cmap='gray')
# class 0
plt.scatter(idx_class0[:,0], idx_class0[:,1], c='green')
# class 1
plt.scatter(idx_class1[0,:,0], idx_class1[0,:,1], c='red')
plt.scatter(idx_class1[1,:,0], idx_class1[1,:,1], c='red')
plt.scatter(idx_class1[2,:,0], idx_class1[2,:,1], c='red')


# ------------------ prepare training features ------------------ #

# stack data to be of size [2xn]
train_X = np.vstack((idx_class0, idx_class1[0], idx_class1[1], idx_class1[2])).transpose() # [2x350]
# scale features
train_X /= 1000
# add bias term
train_X = np.vstack((np.ones(len(train_X[0])),train_X))

# create training labels
train_Y = np.concatenate((np.zeros(idx_class0.shape[0]), 
                          np.ones(idx_class1.shape[0]*idx_class1.shape[1])))
train_Y = train_Y.reshape(1, -1)

# initialize weights
theta = np.random.rand(1,3)


# ------------------ Create model ------------------ #

# logistic function sigmoid
def sigmoid(x):
    return 1./(1+np.exp(-x))

# cross entropy loss function
def corss_entropy_loss(predictions, labels):
    term1 = np.dot(np.log(predictions)[0], labels[0])
    term2 = np.dot(np.log(1-predictions[0]), (1-labels[0]))
    return -1 * np.sum(term1+term2) / len(predictions)
    
# set hyperparameters
lr = 5
epochs = 200
loss = []
class_error_train = []

plt.figure(figsize=(15,5))
# Solution using Gradient Descent Algorithm for Logistic Regression
for e in range(1, epochs+1):
    h = np.matmul(theta, train_X)   # Linear Hypothesis
    Y = sigmoid(h)                  # logistic function to convert the linear hypothesis to logistic regression
    # determine class error
    class_error_train.append(np.sum((Y<0.5)[0]==train_Y[0])/len(train_Y))
    # determine gradient
    grad = np.matmul(train_X,(Y-train_Y).transpose())/len(train_X[0])
    theta = theta - lr*grad.T
    # determine cross entropy loss
    loss.append(corss_entropy_loss(Y, train_Y))

    if e % 10 == 0:  # plotting every 10 iterations
        plt.clf()
        # plotting the cross entropy loss
        plt.subplot(1,2,1)
        plt.plot(loss, color='blue')
        plt.xlabel('iteration no.')
        plt.ylabel('loss value')
        plt.title('Cross Entropy Loss')
        
        # plotting cost function vs iterations
        plt.subplot(1,2,2)
        plt.plot(class_error_train)
        plt.xlabel('iteration no.')
        plt.ylabel('loss')
        plt.title('Classification Error on Training Examples')
        plt.pause(0.2) # change pause time to accommodate for graphing speed
    
    # check for convergence
    if len(loss) > 2:
        convg = abs(loss[-1] - loss[-2]) / loss[-1]
        if convg < lr*1e-4: break


plt.show()