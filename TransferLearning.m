clc
clear all
close all
tuneNetwork = true;

%% load pre-trained modelnet40 network
load('modelnet40PretrainNetwork.mat')

%% upload data and label for transfer learning
if tuneNetwork == true
    pathTrain = '.\datasets\MergeSydney_data\train/';
    dataTrain = dataSydneyReadFuction(pathTrain);
   
    pathValidated = '.\datasets\MergeSydney_data\test/';
    dataValidated = dataSydneyReadFuction(pathValidated);
 
    % load training and testing labels
    load('LabelsMergeTrain.mat')
    traininglabels = categorical(class_sydneyMergeTrain);
    load('LabelsMergeTest.mat')
    validationlabels = categorical(class_sydneyMergeTest);
else
    pathTrain = '.\datasets\sydney_txt_files\train/';
    dataTrain = dataSydneyReadFuction(pathTrain);

    pathValidated = '.\datasets\sydney_txt_files\test/';
    dataValidated = dataSydneyReadFuction(pathValidated);
     
    % load traing and testing labels
    load('LabelsSydneyTrain.mat')
    traininglabels = categorical(class_sydneyTrain);
    load('LabelsSydneyTest.mat')
    validationlabels = categorical(class_sydneyTest);

end

% update layers
numClasses = 3;
layers = voxnet.Layers;
layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(numClasses,'Name','fc4');
layers(end+1) = softmaxLayer('Name','softmax');
layers(end+1) = classificationLayer('Name','crossEntropyLoss');

voxnet_update = layerGraph(layers);

% training options
rng(123) % Set the random seed for reproducing training results
dsLength = length(dataTrain.Files);
if dsLength < 2000
    % Stochastic gradient descent can work just fine (small training set)
    miniBatchSize = dsLength;
    dropPeriod = 10;
else
    % Mini batch gradient descent must be applied (large training set)
    miniBatchSize = 64;
    iterationsPerEpoch = floor(dsLength/miniBatchSize);
    dropPeriod = floor(8000/iterationsPerEpoch);
end

options_2 = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'InitialLearnRate',0.01,...
    'LearnRateSchedule','Piecewise',...
    'LearnRateDropPeriod',dropPeriod,...
    'LearnRateDropFactor', 0.1,...
    'ValidationData',dataValidated,...
    'MaxEpochs',60,...
    'DispatchInBackground',false,...
    'Plots','training-progress',...
    'Shuffle','never');

% train network
voxnet_update = trainNetwork(dataTrain,voxnet_update,options_2);
% save the network
save('voxnet_update.mat','voxnet_update')
 
%% Evaluate network
% outputLabels = classify(voxnet_update,dataValidated);
% accuracy = nnz(outputLabels == validationlabels) / numel(outputLabels);
% disp('ANN classifier:') 
% disp(accuracy)
% 
% %% trainig SVM classifier 
% featureLayer = 'fc1';
% trainingFeatures = activations(voxnet_update, dataTrain, featureLayer, 'OutputAs', 'columns');
% % train classifier
% classifier = fitcecoc(trainingFeatures',traininglabels);
% %% training svm classifier
% % feature extraction
% testingFeatures = activations(voxnet_update, dataValidated, featureLayer, 'OutputAs', 'columns');
% testingFeatures = testingFeatures';
% % classify point clouds
% classifierTest = predict(classifier,testingFeatures);
% 
% accuracy = nnz(classifierTest == validationlabels) / numel(classifierTest);
% disp('SVM classifier:')
% disp(accuracy)

