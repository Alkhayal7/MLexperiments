close all
clear all
clc

data_2d = zeros(1000,1000,3);
idx_class1 = round(rand(175,2)*500+250);
idx_class2 = [round(rand(75,2)*250+50);round(rand(75,2)*350+[500,50]);round(rand(125,2)*150+700)];

% data_2d = insertShape(data_2d,'circle',[idx_class1,5*ones(length(idx_class1),1)],'LineWidth',2,'Color', 'green');
% data_2d = insertShape(data_2d,'circle',[idx_class2,5*ones(length(idx_class2),1)],'LineWidth',2,'Color', 'red');
% imshow(data_2d)
% figure
% return

Xtrain = [idx_class1;idx_class2];   %Training Features
Ytrain = [ones(length(idx_class1),1);ones(length(idx_class2),1)*2];  %Training Labels
% return

s1 = size(data_2d);

for r = 1:1:s1(1)
    r*100/s1(1)
    for c = 1:1:s1(2)
        Xtest = [c,r];  %Test Vector (two features)
%         label = NearestNeighbor(Xtest,Xtrain,Ytrain);%NN
        k = 53;
%         label = KNearestNeighbor(Xtest,Xtrain,Ytrain,k,1);%KNN
%         label = WeightedNearestNeighbor(Xtest,Xtrain,Ytrain);
        if(label==1)
            data_2d(r,c,:)=[0,0.5,0];%Green
        else
            data_2d(r,c,:)=[0.5,0,0];%Red
        end
    end
end

% Mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',k,'Distance','euclidean');%,'Standardize',1);

% figure
data_2d = insertShape(data_2d,'circle',[idx_class1,5*ones(length(idx_class1),1)],'LineWidth',4,'Color', [0.1,1,0]);
data_2d = insertShape(data_2d,'circle',[idx_class2,5*ones(length(idx_class2),1)],'LineWidth',4,'Color', [1,0,0]);
imshow(data_2d)


function [Ytest] = NearestNeighbor(Xtest,Xtrain,Ytrain)
%Vectorized Code (No for loops, faster)
% tic
diff = Xtest - Xtrain;
diffSq = diff.^2;
L2 = sum(diffSq,2);
[m,i] = min(L2);
Ytest = Ytrain(i);

% L1 = sum(abs(diff),2);

% toc

%for loops based code, slower
% tic
% s = size(Xtrain);
% minD = inf;
%     for i = 1:s(1)
%         diff = Xtest - Xtrain(i,:);
%         diffSq = diff.^2;
%         L2 = sum(diffSq);
%         if(L2<minD)
%             Ytest = Ytrain(i);
%             minD = L2;
%         end
%     end
%     toc
end

function [Ytest,mindist] = KNearestNeighbor(Xtest,Xtrain,Ytrain,k,distMetric)
%Vectorized Code (No for loops, faster)
% tic
diff = Xtest - Xtrain;
if(distMetric==1)
    d = sum(abs(diff),2); %L1
else
    d = sum(diff.^2,2); %L2
end
[m,idx] = sort(d);     %Sort all the scores
Ytest = mode(Ytrain(idx(1:k))); %Find mode of the 'k' training examples with minimum distance
mindist = d(idx(1));
end

function [Ytest] = WeightedNearestNeighbor(Xtest,Xtrain,Ytrain,k)
diff = Xtest - Xtrain;
% L1 = sum(abs(diff),2);
L2 = sum(diff.^2,2);
dist = L2;
c1 = dist(Ytrain==1);
c2 = dist(Ytrain==2);
c1 = sum(1./c1);
c2 = sum(1./c2);
if(c1<c2)
    Ytest = 2;
else
    Ytest = 1;
end
end
