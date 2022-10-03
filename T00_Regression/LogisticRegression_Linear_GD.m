close all
clear all
clc

data_2d = zeros(1000,1000,3);%3 primary colors, Red, Green and Blue
rng(1);% Set random generator to give same random numbers every time this code is run.
idx_class0 = round(rand(125,2)*150+700); %random numbers between 700 and 850
idx_class1 = [round(rand(75,2)*250+50);round(rand(75,2)*350+[500,50]);round(rand(75,2)*350+[50,500])];

data_2d = insertShape(data_2d,'circle',[idx_class0,5*ones(length(idx_class0),1)],'LineWidth',2,'Color', 'green');
data_2d = insertShape(data_2d,'circle',[idx_class1,5*ones(length(idx_class1),1)],'LineWidth',2,'Color', 'red');

figure
imshow(data_2d)
title('Two Class Problem - Training Examples')
pause(1)
figure
% return

trainX = [idx_class0;idx_class1]';
trainX = trainX/1000; %feature scaling
trainX = [ones(1,length(trainX));trainX];
trainY = [zeros(length(idx_class0),1);ones(length(idx_class1),1)]';

theta = rand(1,3);

lr = 5.75;%A large value of learning rate is possible because features have been properly scaled and dataset is simple. 
loss = [];
ClassErrorTrain = [];
cnt = 0;

while(1)
    h = theta*trainX;%Linear Hypothesis
    Y = 1./(1+exp(-h));%logistic function to convert the linear hypothesis to logistic regression
    ClassErrorTrain = [ClassErrorTrain sum((Y<0.5)==trainY)/length(trainY)];
    Grad = (trainX*(Y-trainY)')/length(trainX);
    theta = theta-lr*Grad';
    loss = [loss -sum(log(Y).*trainY + log(1-Y).*(1-trainY))/length(trainX)];%Cross Entropy
    if(rem(cnt,2)==0)
        subplot(1,2,1)
        plot(loss)
        title('Cross Entropy Loss')
        subplot(1,2,2)
        plot(ClassErrorTrain)
        title('Classification Error on Training Examples')
        drawnow
    end
%     pause(0.1)
    cnt = cnt + 1;
    if(cnt>200)
        break;
    end
    if(cnt>10) %Run for at least 10 iterations
        termCond = abs(loss(end)-loss(end-1))/loss(end);
        if(termCond<1e-9) %Stop if loss decreases by less than 1% 
            break;
        end
    end
end
% pause

s = size(data_2d);
data_2d_indx = ones(1000,1000);
[testXc,testXr] = find(data_2d_indx);
testXc = testXc/1000;
testXr = testXr/1000;
% testX = [ones(1,1e6);testXc';testXr'];
testX = [ones(1,1e6);testXc';testXr'];
h = theta*testX;%Linear Hypothesis
labels = 1./(1+exp(-h));

i=1;
for r = 1:1:s(1)
%     r
    for c = 1:1:s(2)
%         label = hyp_lr([1;c;r],theta);
        label = labels(i);
        i = i+1;
        if(label<0.5)
            data_2d(r,c,:)=[0,0.5,0];%Green
        else
            data_2d(r,c,:)=[0.5,0,0];%Red
        end
    end
end
data_2d = insertShape(data_2d,'circle',[idx_class0,5*ones(length(idx_class0),1)],'LineWidth',2,'Color', 'green');
data_2d = insertShape(data_2d,'circle',[idx_class1,5*ones(length(idx_class1),1)],'LineWidth',2,'Color', 'red');

figure
imshow(data_2d)
title('Classification using the trained model')
