%% LSTM
%  clc
close all
clear all

% load features
load ('FeaturesTrain.mat')
%  introduce delay
Delay = 1;
XTrain = trainingFeatures(:,1:end-Delay);
YTrain = trainingFeatures(:,Delay+1:end);
numFeatures = 128;                       % S x D where D is the number of features and S is time stamps 
numResponses = 128;                      % size of output vector or in case of categorical data it should be equal to total number of categories
numHiddenUnits = 100;                    % Can be changed to 300 or 400   (h- hidden states in lstm) 

% Network layers 
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 100;                        % Iterations to converge the accuracy and loss
miniBatchSize = 32;                     % Size of mini-batches use for prediction, specified as a positive integer. Larger mini-batch sizes require more memory, but can lead to faster predictions.

% Training Options
options = trainingOptions('sgdm', ...   % Stochastic gradient descent with momentum ; Use the adam optimizer. You can specify the decay rates of the gradient and squared gradient moving averages using the 'GradientDecayFactor' and 'SquaredGradientDecayFactor' name-value pair arguments
    'ExecutionEnvironment','cpu', ...   % Instructions for using cpu, we can change it for gpu
    'GradientThreshold',1, ...          % If the gradient exceeds the value of GradientThreshold, then the gradient is clipped 
    'MaxEpochs',maxEpochs, ...          % Iterations , in a single epoch ,full data passes once
    'MiniBatchSize',miniBatchSize, ...  % A mini-batch is a subset of the training set that is used to evaluate the gradient of the loss function and update the weights.
    'SequenceLength','longest', ...     % Option to pad, truncate, or split input sequences
    'Shuffle','every-epoch', ...        % Option for data shuffling (once, never, every-epoch)
    'Verbose',1, ...                    % Indicator to display training progress information
    'Plots','training-progress');       % plot shows when the training is in progress


% % Train the LSTM Network 

net = trainNetwork(XTrain,YTrain,layers,options);  


net = predictAndUpdateState(net,XTrain);   


[net,YPred] = predictAndUpdateState(net,YTrain);

% plot predicting features
plot(YPred')

% load testing features
load('FeaturesTest.mat')
numTimeStepsTest = length(testingFeatures);

for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,testingFeatures(:,i-1),'ExecutionEnvironment','cpu');
end
figure;
plot(YPred')
title('YPred')
% MSE = sqrt(mean((YPred-YTest).^2));
Difference = abs((YPred-testingFeatures)./testingFeatures);
MAPE = (median(Difference))*100;

display(['Minimum: ', num2str(min(MAPE))])
display(['Maximum: ', num2str(max(MAPE))])
display(['Mean: ', num2str(mean(MAPE))])
display(['Median: ', num2str(median(MAPE))])