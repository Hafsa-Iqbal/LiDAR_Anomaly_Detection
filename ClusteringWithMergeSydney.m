%% INSTRUCTIONS
% take one txt file from the iCAB dataset at a time
% change "clusterNumber" to change the number of cluster inside
% "dataPC_clustering" file. 
% Quantitative/Qualitative analysis can be done by observing that the 
% prediction from the "classifierTest" is same as shown in  figure which
% contain the voxel representation of cluster.



clc
clear all
close all

%% load pre-trained classification network
cd ('C:\Users\hafsa\OneDrive\Desktop');
load('modelnet40PretrainNetwork.mat')
%% upload merge data

path = 'C:\Users\hafsa\OneDrive\Desktop\sydney-urban-objects-dataset\MergeData_sydney_txt_files';
pcds = dataSydneyReadFuction(path);   % augmentation and voxelization
dataTrain =  pcds ;
dataPreview = preview(pcds);
%% plot  voxels
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


%% load sabatini building data for the feature extraction
 path = 'C:\Users\hafsa\OneDrive\Desktop\TEMPORARY_pc';  % use one file at a time
% data handling, pre-processing and conversion 
grid_vox = 32;
[pcds]= dataPC_clustering(path, grid_vox);   
testingSet = pcds;

%% plot point cloud/ voxels
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
classifierTest = predict(classifier,testingFeatures')


