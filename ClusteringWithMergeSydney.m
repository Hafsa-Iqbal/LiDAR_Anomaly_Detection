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

%% load clustered txt files
path = 'C:\Users\hafsa\OneDrive\Desktop\clusteredData\1';  % clustered in format of txt files
% data handling and voxelization conversion 
grid_vox = 32;
pcds = dataPC_clustering(path,grid_vox); 
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
classifierTest = predict(classifier,testingFeatures')
