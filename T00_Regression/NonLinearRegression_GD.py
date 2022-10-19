# Author: Ahmed Alkhayal - EE482
import numpy as np
import matplotlib.pyplot as plt

# setting seed 
np.random.seed(0)

# ----------- generating data data ------------------ #

# Generate artificial training examples
x = np.arange(-10, 10, 0.1)                      # one feature
y = 50*np.random.rand(len(x)) + 3*x + 0.5 * x**3 # labels

# adding noise to labels
y = y + np.random.rand(y.shape[0]) * 10

# plotting the scatter plot of features and corresponding labels
plt.figure(figsize=(10,5))
plt.scatter(x, y)
plt.xlabel('X Values (Feature)')
plt.ylabel('Y Values (Label)')

# ----------- scaling data ------------------ #
max_value = np.max(np.abs(x))
scaled_x = x/max_value

# ----------- preparing algorithm data ------------------ #

# features of training examples
train_x = np.stack((np.ones_like(scaled_x), #bias
                    scaled_x,               #x
                    scaled_x**2,            #x^2    
                    scaled_x**3))           #x^3        
train_y = y                                             # labels of training examples
n = len(train_x[0])                                     # no. of training examples
theta = np.random.rand(1,train_x.shape[0])              # initial random weights
lr = 0.1                                                # learning rate
loss = []                                               # track loss over iterations


# ----------- training algorithm ------------------ #

plt.figure(figsize=(10,5))     
# Solution using Gradient Descent Algorithm for Linear Regression
iter = 0
while(iter < 1e4):
    iter += 1
    h = np.matmul(theta, train_x)               # current hypothesis
    j = np.sum((h-train_y)**2) / (2*n)          # cost function
    dj = np.matmul(train_x, (h-train_y).T) / n  # partial gradients of Cost function using vectorized code
    theta = theta - lr * dj.T                   # update theta
    loss.append(j)                              # loss/cost history for plotting
    
    if iter % 10 == 0:  # plotting every 10 iterations
        plt.clf()       # clear figure
        
        # plotting Linear regression line
        plt.subplot(1,2,1)
        plt.scatter(x, y, color='red', marker='.')
        plt.plot(x, h[0], color='blue')
        plt.ylabel('x (feature)')
        plt.xlabel('y (label)')
        plt.title('Linear regression line')

        # plotting cost function vs iterations
        plt.subplot(1,2,2)
        plt.plot(loss)
        plt.ylabel('Loss / Cost')
        plt.xlabel('iteration no.')
        plt.title('Cost function vs. iterations')
        plt.pause(0.1) # change pause time to accommodate for graphing speed
        
    # check for convergence
    if len(loss) > 2:
        convg = abs(loss[-1] - loss[-2]) / loss[-1]
        if convg < lr*1e-3: break

print(theta)
# keep figures alive after execution
plt.show()
