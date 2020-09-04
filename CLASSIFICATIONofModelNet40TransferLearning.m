%% pretrainedNetwork is sydney here while for transfer learning modelnet40 is used
clc
clear all
% close all
% 
%% load pre-trained classification network
cd ('C:\Users\hafsa\OneDrive\Desktop');
load('sydneyPreTrainedNetwrok.mat');
%% load modelnet40 data
path_train = 'C:\Users\hafsa\OneDrive\Desktop\Matlab Files\train/';
path_test = 'C:\Users\hafsa\OneDrive\Desktop\Matlab Files\test/';

[pcds_train] = dataModelnet40(path_train);
[pcds_test] = dataModelnet40(path_test);
%% plot
temp = preview(pcds_train);
figure
p = patch(isosurface(temp,0.5));
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(45,45)
camlight;
lighting phong

% feature extraction layer
featureLayer = 'fc1';

% perfrom feature extraction
trainingFeatures = activations(voxnet, pcds_train, featureLayer, 'OutputAs', 'columns');

% labels of training data
load('labelsTrain.mat')
labelstrain = categorical(class_train);
% train classifier
trainingLabels = labelstrain;
classifier = fitcecoc(trainingFeatures',labelstrain);




%%
testingSet = pcds_test;
% feature extraction
testingFeatures = activations(voxnet, testingSet, featureLayer, 'OutputAs', 'columns');
% classify point clouds
classifierTest = predict(classifier,testingFeatures');

load('labels.mat')
labelsTest = categorical(class1);
accuracy = nnz(classifierTest  == labelsTest) / numel(labelsTest);
disp(accuracy)
% FEATURE PLOTS
testingFeatures = testingFeatures';
figure;
Y_testingFeatures = tsne(testingFeatures);
gscatter(Y_testingFeatures(:,1),Y_testingFeatures(:,2),classifierTest)

%%

function [pcds]= dataModelnet40(path)

pcds = imageDatastore(path,...
    'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

    function Voxel_Data = pc_reader(filename)
        grid_vox = 32;
        n_points = 2048;
        points = table2array(readtable(filename));
        points = points(randperm(n_points),:);
        points = Shrink2UnitSphere((points));
        points = pointCloud(points);
        indices_occupancy = pcbin(points,[grid_vox grid_vox grid_vox]);
        occupancyGrid = cellfun(@(c) ~isempty(c),indices_occupancy);
        Voxel_Data = occupancyGrid;
    end



end

function [newPoints]=Shrink2UnitSphere(Points)
%Shrink2UnitSphere shrinks the given data x,y,z Points to fit inside a unit sphere
%and returns the new dataset
%INPUT : Points nx3
% move model to center of geometry
xyzmean = mean(Points, 1);
newPoints(:, 1) = Points(:, 1) - xyzmean(1);
newPoints(:, 2)=Points(:, 2) - xyzmean(2);
newPoints(:, 3)=Points(:, 3) - xyzmean(3);
%Shrink the model
dist = sqrt(sum(newPoints.^2,2));
maxdist = max(dist);
newPoints = newPoints/(maxdist);
end
