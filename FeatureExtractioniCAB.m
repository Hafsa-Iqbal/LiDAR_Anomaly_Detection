% clc
clear all
close all
tuneNetwork = true;

% upload data and label for transfer learning
if tuneNetwork == true
   % load pre-trained voxnet_update network after transfer learning
    load('voxnet_update.mat')
else
     % load pre-trained modelnet40 network
    load('modelnet40PretrainNetwork.mat')
end

% testing data to extract features
path = '.\datasets\iCAB_data';
grid_vox = 32;
% data handling, pre-processing, clustering, selection of nearest cluster and voxelization
testingdataset = dataPC_clustering(path,grid_vox) ;

% plot voxel
dataread = preview(testingdataset);
figure
p = patch(isosurface(dataread,0.5));
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(45,45)
camlight;
lighting phong

% feature extraction
featureLayer = 'fc1';
trainingFeatures = activations(voxnet_update, testingdataset, featureLayer, 'OutputAs', 'columns');

save('Features.mat','trainingFeatures')
