 clc
clear all
close all

%% load pre-trained modelnet40 network to extract feature
load('modelnet40PretrainNetwork.mat')
%% upload merge data
path = '.\datasets\MergeSydney_data/';
pcds = dataSydneyReadFuction(path);   % augmentation and voxelization
dataTrain =  pcds ;

%% plot  voxels
% dataPreview = preview(pcds);
% figure
% p = patch(isosurface(dataPreview,0.5));
% p.FaceColor = 'red';
% p.EdgeColor = 'none';
% daspect([1 1 1])
% view(45,45)
% camlight; 
% lighting phong

% load labels 
load('LabelsSydney.mat')
traininglabels = categorical(class_sydney);

% feature extraction layer
featureLayer = 'fc1';

% perform feature extraction
trainingFeatures = activations(voxnet, dataTrain, featureLayer, 'OutputAs', 'columns');

% train classifier
classifier = fitcecoc(trainingFeatures',traininglabels);

%% load clustered txt files
path = '.\datasets\iCAB data';  
% data handling, ore-processing and voxelization 
grid_vox = 32;
pcds = dataPC(path,grid_vox); 
testingSet = pcds;

%% plot voxels
dataPreview = preview(testingSet);
figure
p = patch(isosurface(dataPreview,0.5));
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(45,45)
camlight; 
lighting phong
%%
% feature extraction
testingFeatures = activations(voxnet, testingSet, featureLayer, 'OutputAs', 'columns');
% classify point clouds
classifierTest = predict(classifier,testingFeatures');
% save features
save('Features.mat','testingFeatures')
