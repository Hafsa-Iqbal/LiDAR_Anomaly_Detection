% clc
clear all
close all
tuneNetwork = true;

%% load pre-trained modelnet40 network 
load('modelnet40PretrainNetwork.mat')

%% upload data and label for transfer learning
if tuneNetwork == true
    pathTrain = '.\datasets\MergeSydney_data\train/';
    pcds = dataSydneyReadFuction(pathTrain);  
    dataTrain =  pcds ;
    
    pathValidated = '.\datasets\MergeSydney_data\test/';
    pcds = dataSydneyReadFuction(pathValidated);   
    dataValidated =  pcds ;
    % load training and testing labels
    load('LabelsMergeTrain.mat')
    traininglabels = categorical(class_sydneyMergeTrain);
    load('LabelsMergeTest.mat')
    validationlabels = categorical(class_sydneyMergeTest);
else
    pathTrain = '.\datasets\sydney_txt_files\train/'; 
    pcds = dataSydneyReadFuction(pathTrain);   
    dataTrain =  pcds ;
    
    pathValidated = '.\datasets\sydney_txt_files\test/';
    pcds = dataSydneyReadFuction(pathValidated);  
    dataValidated =  pcds ;
    % load traing and testing labels
    load('LabelsSydneyTrain.mat')
    traininglabels = categorical(class_sydneyTrain);
    load('LabelsSydneyTest.mat')
    validationlabels = categorical(class_sydneyTest);
    
end

% update layers 
numFeatures = 1024;
numClasses = 3;
layers = voxnet.Layers;
layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(numFeatures,'Name','fc2');
layers(end+1) = reluLayer('Name','relu1');
layers(end+1) = fullyConnectedLayer(numClasses,'Name','fc3');
layers(end+1) = softmaxLayer('Name','softmax');
layers(end+1) = classificationLayer('Name','crossEntropyLoss');

voxnet_update = layerGraph(layers);

% training options
miniBatchSize = 32;
dsLength = length(dataTrain.Files);
iterationsPerEpoch = floor(dsLength/miniBatchSize);
dropPeriod = floor(8000/iterationsPerEpoch);

options_2 = trainingOptions('sgdm','InitialLearnRate',0.01,'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','Piecewise',...
    'LearnRateDropPeriod',dropPeriod,...
    'ValidationData',dataValidated,'MaxEpochs',60,...
    'DispatchInBackground',false,...
    'Plots','training-progress',...
    'Shuffle','never');

% train network
voxnet_update = trainNetwork(dataTrain,voxnet_update,options_2);

% Evaluate network

outputLabels = classify(voxnet_update,dataValidated);
accuracy = nnz(outputLabels == validationlabels) / numel(outputLabels);
disp(accuracy)



%% load icab data 
path = '.\datasets\iCAB_data';
% data handling, pre-processing and voxelization
grid_vox = 32;
pcds = dataPC(path,grid_vox);
testingdataset = pcds;

%% plot voxels
dataPreview = preview(testingdataset);
figure
p = patch(isosurface(dataPreview,0.5));
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(45,45)
camlight;
lighting phong
%% feature extraction 

% feature extraction layer
featureLayer = 'fc2';
% feature extraction
testingFeatures = activations(voxnet_update, testingdataset, featureLayer, 'OutputAs', 'columns');

% save features
save('Features.mat','testingFeatures')
