%% Pre-train Network for ModelNet40
clear all

path_train = '.\datasets\train/';
path_test = '.\datasets\test/';

[pcds_train] = dataModelnet40(path_train);
[pcds_test] = dataModelnet40(path_test);

%% Plot
dataPreview = preview(pcds_train);
figure
p = patch(isosurface(dataPreview,0.5));
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(45,45)
camlight;
lighting phong

%% Define network layers
numClasses = 40;
layers = [image3dInputLayer([32 32 32],'Name','inputLayer','Normalization','none'),...
    convolution3dLayer(5,32,'Stride',2,'Name','Conv1'),...
    leakyReluLayer(0.1,'Name','leakyRelu1'),...
    convolution3dLayer(3,32,'Stride',1,'Name','Conv2'),...
    leakyReluLayer(0.1,'Name','leakyRulu2'),...
    maxPooling3dLayer(2,'Stride',2,'Name','maxPool'),...
    fullyConnectedLayer(128,'Name','fc1'),...
    reluLayer('Name','relu'),...
    dropoutLayer(0.5,'Name','dropout1'),...
    fullyConnectedLayer(numClasses,'Name','fc2'),...
    softmaxLayer('Name','softmax'),...
    classificationLayer('Name','crossEntropyLoss')];

voxnet = layerGraph(layers);
figure
plot(voxnet);

% Setup training options
rng(123) % Set a the random seed for reproducing training results
max_epoch = 300;
MiniBatchSize = 128;
dsLength = length(pcds_train.Files);
ValidationFrequency = uint64(5*dsLength/MiniBatchSize); 
InitialLearnRate = 0.001;
LearnRateSchedule = 'piecewise';
LearnRateDropPeriod = 15;
LearnRateDropFactor = 0.7;
DispatchInBackground = true;

 options = trainingOptions('sgdm', ...
    'MaxEpochs',max_epoch, ...
    'ValidationData',pcds_test, ...
    'ValidationFrequency',ValidationFrequency, ...
    'Verbose',false, ...
    'MiniBatchSize', MiniBatchSize,...
    'InitialLearnRate', InitialLearnRate,...
    'LearnRateSchedule', LearnRateSchedule,...
    'LearnRateDropPeriod', LearnRateDropPeriod,...
    'LearnRateDropFactor', LearnRateDropFactor,...
    'DispatchInBackground',DispatchInBackground,...
    'Shuffle', 'every-epoch',...
    'Plots','training-progress');

% Train network
voxnet = trainNetwork(pcds_train,voxnet,options);

% save trained network
save('modelnet40PretrainNetwork.mat',voxnet)

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
        % convert vector into point cloud
        points = pointCloud(points);   
        % point cloud to voxelization
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
