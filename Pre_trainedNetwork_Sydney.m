%% pre- trained network for sydney urban data
clc
clear all
close all

threeClasses = true; 

dataPath = 'C:\Users\hafsa\OneDrive\Desktop\sydney-urban-objects-dataset';
% dataPath = downloadSydneyUrbanObjects(tempdir);   % you can directly download sydney data using that function
dsTrain = sydneyUrbanObjectsClassificationDatastore(dataPath,[1 2 3]);
dsVal = sydneyUrbanObjectsClassificationDatastore(dataPath,4);

% plot labels
dsLabels = transform(dsTrain,@(data) data{2});
labels = readall(dsLabels);
figure
histogram(labels)

dsLabelsVAL = transform(dsVal,@(data) data{2});
labelsVAL = readall(dsLabelsVAL);
figure
histogram(labelsVAL)

%% for three classes  (pedistrain, tree, building)
if threeClasses == true 
    % extract data relevant to three classes 
    indexTree = find(labels == 'tree');
    indexPedestrain = find(labels == 'pedestrian');
    indexBuilding = find(labels == 'building');
    IndexTreePedestrainBuilding = sort([indexTree;indexPedestrain;indexBuilding]);
    dsTrain.Files = dsTrain.Files(IndexTreePedestrainBuilding);
    
    indexTree = find(labelsVAL == 'tree');
    indexPedestrain = find(labelsVAL == 'pedestrian');
    indexBuilding = find(labelsVAL == 'building');
    IndexTreePedestrainBuilding = sort([indexTree;indexPedestrain;indexBuilding]);
    dsVal.Files = dsVal.Files(IndexTreePedestrainBuilding);
    
    % update labels of traning and validated data and plot histogram
    dsLabels = transform(dsTrain,@(data) data{2});
    labels = readall(dsLabels);
    figure
    histogram(labels)
    
    dsLabelsVAL = transform(dsVal,@(data) data{2});
    labelsVAL = readall(dsLabelsVAL);
    figure
    histogram(labelsVAL)
end
%%  
% Data augmentation pipeline
dsTrain = transform(dsTrain,@augmentPointCloudData);

% plot point cloud and label it
dataOut = preview(dsTrain);
figure
pcshow(dataOut{1});
title(dataOut{2});
dataOut{1}.Count
dataOut{2}

% point cloud to voxelization
dsTrain = transform(dsTrain,@formOccupancyGrid);
dsVal = transform(dsVal,@formOccupancyGrid);


layers = [image3dInputLayer([32 32 32],'Name','inputLayer','Normalization','none'),...
    convolution3dLayer(5,32,'Stride',2,'Name','Conv1'),...
    leakyReluLayer(0.1,'Name','leakyRelu1'),...
    convolution3dLayer(3,32,'Stride',1,'Name','Conv2'),...
    leakyReluLayer(0.1,'Name','leakyRulu2'),...
    maxPooling3dLayer(2,'Stride',2,'Name','maxPool'),...
    fullyConnectedLayer(128,'Name','fc1'),...
    reluLayer('Name','relu'),...
    dropoutLayer(0.5,'Name','dropout1'),...
    fullyConnectedLayer(14,'Name','fc2'),...
    softmaxLayer('Name','softmax'),...
    classificationLayer('Name','crossEntropyLoss')]; 

voxnet = layerGraph(layers);
% plot network layers
figure
plot(voxnet);

% Setup training options
miniBatchSize = 32;
dsLength = length(dsTrain.UnderlyingDatastore.Files);
iterationsPerEpoch = floor(dsLength/miniBatchSize);
dropPeriod = floor(8000/iterationsPerEpoch);

options = trainingOptions('sgdm','InitialLearnRate',0.01,'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','Piecewise',...
    'LearnRateDropPeriod',dropPeriod,...
    'ValidationData',dsVal,'MaxEpochs',60,...
    'DispatchInBackground',false,...
    'Plots','training-progress',...
    'Shuffle','never');

% Train network
voxnet = trainNetwork(dsTrain,voxnet,options);

% Evaluate network
valLabelSet = transform(dsVal,@(data) data{2});
valLabels = readall(valLabelSet);
outputLabels = classify(voxnet,dsVal);
accuracy = nnz(outputLabels == valLabels) / numel(outputLabels);
disp(accuracy)

% confusion matrix 
figure
plotconfusion(valLabels,outputLabels)

save('sydneyPreTrainedNetwrok.mat','voxnet')
function datasetPath = downloadSydneyUrbanObjects(dataLoc)

if nargin == 0
    dataLoc = pwd();
end

dataLoc = string(dataLoc);

url = "http://www.acfr.usyd.edu.au/papers/data/";
name = "sydney-urban-objects-dataset.tar.gz";

if ~exist(fullfile(dataLoc,'sydney-urban-objects-dataset'),'dir')
    disp('Downloading Sydney Urban Objects Dataset...');
    untar(fullfile(url,name),dataLoc);
end

datasetPath = dataLoc.append('sydney-urban-objects-dataset');

end

function ds = sydneyUrbanObjectsClassificationDatastore(datapath,folds)
% sydneyUrbanObjectsClassificationDatastore Datastore with point clouds and
% associated categorical labels for Sydney Urban Objects dataset.
%
% ds = sydneyUrbanObjectsDatastore(datapath) constructs a datastore that
% represents point clouds and associated categories for the Sydney Urban
% Objects dataset. The input, datapath, is a string or char array which
% represents the path to the root directory of the Sydney Urban Objects
% Dataset.
%
% ds = sydneyUrbanObjectsDatastore(___,folds) optionally allows
% specification of desired folds that you wish to be included in the
% output ds. For example, [1 2 4] specifies that you want the first,
% second, and fourth folds of the Dataset. Default: [1 2 3 4].

if nargin < 2
    folds = 1:4;
end

datapath = string(datapath);
path = fullfile(datapath,'objects',filesep);

% For now, include all folds in Datastore
foldNames{1} = importdata(fullfile(datapath,'folds','fold0.txt'));
foldNames{2} = importdata(fullfile(datapath,'folds','fold1.txt'));
foldNames{3} = importdata(fullfile(datapath,'folds','fold2.txt'));
foldNames{4} = importdata(fullfile(datapath,'folds','fold3.txt'));
names = foldNames(folds);
names = vertcat(names{:});

fullFilenames = append(path,names);
ds = fileDatastore(fullFilenames,'ReadFcn',@extractTrainingData,'FileExtensions','.bin');

% Shuffle
ds.Files = ds.Files(randperm(length(ds.Files)));

end

function dataOut = extractTrainingData(fname)

[pointData,intensity] = readbin(fname);

[~,name] = fileparts(fname);
name = string(name);
name = extractBefore(name,'.');
name = replace(name,'_',' ');

labelNames = ["4wd","building","bus","car","pedestrian","pillar",...
    "pole","traffic lights","traffic sign","tree","truck","trunk","ute","van"];

label = categorical(name,labelNames);

dataOut = {pointCloud(pointData,'Intensity',intensity),label};

end

function [pointData,intensity] = readbin(fname)
% readbin function, Read point and intensity data from Sydney Urban Object binary
% files.

fid = fopen(fname, 'r');
c = onCleanup(@() fclose(fid));

fseek(fid,10,-1); % Move to the first X point location 10 bytes from beginning
X = fread(fid,inf,'single',30);
fseek(fid,14,-1);
Y = fread(fid,inf,'single',30);
fseek(fid,18,-1);
Z = fread(fid,inf,'single',30);

fseek(fid,8,-1);
intensity = fread(fid,inf,'uint8',33);

pointData = [X,Y,Z];

end

function dataOut = formOccupancyGrid(data)

grid = pcbin(data{1},[32 32 32]);
occupancyGrid = zeros(size(grid),'single');
for ii = 1:numel(grid)
    occupancyGrid(ii) = ~isempty(grid{ii});
end
label = data{2};
dataOut = {occupancyGrid,label};

end

function dataOut = augmentPointCloudData(data)

ptCloud = data{1};
label = data{2};

% Apply randomized rotation about Z axis.
tform = randomAffine3d('Rotation',@() deal([0 0 1],360*rand),'Scale',[0.98,1.02],'XReflection',true,'YReflection',true); % Randomized rotation about z axis
ptCloud = pctransform(ptCloud,tform);

% Apply jitter to each point in point cloud
amountOfJitter = 0.01;
numPoints = size(ptCloud.Location,1);
D = zeros(size(ptCloud.Location),'like',ptCloud.Location);
D(:,1) = diff(ptCloud.XLimits)*rand(numPoints,1);
D(:,2) = diff(ptCloud.YLimits)*rand(numPoints,1);
D(:,3) = diff(ptCloud.ZLimits)*rand(numPoints,1);
D = amountOfJitter.*D;
ptCloud = pctransform(ptCloud,D);

dataOut = {ptCloud,label};

end

