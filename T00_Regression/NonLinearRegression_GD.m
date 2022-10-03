close all
clear all
clc

rng(0);%pseudo-random generator seed

% Generate artificial training examples
x = [-10:0.1:10];   %one feature
y = 0.5*x.^3+x*3-5+50*randn(size(x));    %labels
y = y + randn(size(y))*10;
% plot(x,y)
scatter(x,y)
xlabel('X Values (Feature)')
ylabel('Y Values (Label)')

% Scaling is important for convergence. Otherwise, a very small learning
% rate is required which may take very long to converge. 
MaxValue = max(x,[],'all');
xScaled = x/MaxValue;
trainX = [ones(size(xScaled));xScaled;xScaled.^2;xScaled.^3]; %Use higher degree polynomial featuers of training examples 
trainY = y;                     %labels of training examples
n = length(trainX);             %no. of training examples
theta = rand(1,4);
lr = 0.1;                     %learning rate
loss = [];

iter = 0;
%Solution using Gradient Descent Algorithm for Linear Regression
while(1)
    iter = iter + 1;
    h = theta*trainX;               %current hypothesis
    J = sum((h-trainY).^2)/(2*n);   %Cost function (MSE)
    dJ = (trainX*(h-trainY)')/n;    %partial gradients of Cost function using vectorized code
    theta = theta - lr*dJ';         %theta update 
    loss = [loss,J];                %loss/cost history for plotting
    
    if(rem(iter,10)==0) %Plot every 10 iterations only
        subplot(1,2,1)
        scatter(x,y)
        hold on
        plot(x,h)
        hold off
        ylabel('x (feature)')
        xlabel('y (label)')
        title('Non-Linear regression line')
        subplot(1,2,2)
        plot(loss)
        ylabel('Loss / Cost')
        xlabel('iteration no.')
        title('Cost function vs. iterations')
        drawnow
    %     pause(0.5)
    end
    if(length(loss)>2)
    convg = abs(loss(end)-loss(end-1))/loss(end);
        if(convg<lr*1e-3)
            break;
        end
    end
    if(iter>1e4)
        break;
    end
end

