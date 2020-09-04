%% pretrained network is modelnet40 here and transfer learning is used with sydney 
clc
clear all
% close all
threeClasses = true;
%% load pre-trained classification network
cd ('C:\Users\hafsa\OneDrive\Desktop');
load('modelnet40PretrainNetwork.mat')
%%  load sydney data
dataPath = 'C:\Users\hafsa\OneDrive\Desktop\sydney-urban-objects-dataset';
% dataPath = downloadSydneyUrbanObjects(tempdir);
dsTrain = sydneyUrbanObjectsClassificationDatastore(dataPath,[1 2 3]);
dsVal = sydneyUrbanObjectsClassificationDatastore(dataPath,4);

dsLabels = transform(dsTrain,@(data) data{2});
labels = readall(dsLabels);

% Data augmentation pipeline
dsTrain = transform(dsTrain,@augmentPointCloudData);

% voxelization transform to each input point cloud
dsTrain = transform(dsTrain,@formOccupancyGrid);
dsVal = transform(dsVal,@formOccupancyGrid);


dsLabelstrain = transform(dsTrain,@(data) data{2});
labelstrain = readall(dsLabelstrain);


dsLabelsVAL = transform(dsVal,@(data) data{2});
labelsVAL = readall(dsLabelsVAL);

%%
if threeClasses == true
    % extract the three classes from training data
    indexTree = find(labelstrain == 'tree');
    indexPedestrain = find(labelstrain == 'pedestrian');
    indexBuilding = find(labelstrain == 'building');
    IndexTreePedestrainBuilding = sort([indexTree;indexPedestrain;indexBuilding]);
    dsTrain.UnderlyingDatastore.Files = dsTrain.UnderlyingDatastore.Files(IndexTreePedestrainBuilding);
    
     % extract the three classes from validated data
    indexTree = find(labelsVAL == 'tree');
    indexPedestrain = find(labelsVAL == 'pedestrian');
    indexBuilding = find(labelsVAL == 'building');
    IndexTreePedestrainBuilding = sort([indexTree;indexPedestrain;indexBuilding]);
    dsVal.UnderlyingDatastore.Files = dsVal.UnderlyingDatastore.Files(IndexTreePedestrainBuilding);
    
    % update the labels of traing and validated data
    dsLabelstrain = transform(dsTrain,@(data) data{2});
    labelstrain = readall(dsLabelstrain);
    
    dsLabelsVAL = transform(dsVal,@(data) data{2});
    labelsVAL = readall(dsLabelsVAL);
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% feature extraction layer
featureLayer = 'fc1';

% perfrom feature extraction
trainingFeatures = activations(voxnet, dsTrain, featureLayer, 'OutputAs', 'columns');

% train classifier
trainingLabels = labelstrain;
classifier = fitcecoc(trainingFeatures',labelstrain);




%%
testingSet = dsVal;
% feature extraction
testingFeatures = activations(voxnet, testingSet, featureLayer, 'OutputAs', 'columns');
% classify point clouds
classifierTest = predict(classifier,testingFeatures');

accuracy = nnz(classifierTest  == labelsVAL) / numel(labelsVAL);
disp(accuracy)
% FEATURE PLOTS
testingFeatures = testingFeatures';
figure;
Y_testingFeatures = tsne(testingFeatures);
gscatter(Y_testingFeatures(:,1),Y_testingFeatures(:,2),classifierTest)

%% save workspace as
%classificationICAB3Classes_pc.mat
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
% Objects dataset.
%
% ds = sydneyUrbanObjectsDatastore(___,folds) optionally allows
% specification of desired folds that you wish to be included in the
% output ds. 
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
ds = imageDatastore(fullFilenames,'ReadFcn',@extractTrainingData,'FileExtensions','.bin');

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
% readbin Read point and intensity data from Sydney Urban Object binary files.


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



