function [pcds]= dataSydneyReadFuction(F) 

pcds = imageDatastore(F,'ReadFcn',@pc_reader,...   % store data 
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');




function Voxel_Data  = pc_reader(filename)           % read function
pc = table2array(readtable(filename));               % table conversion
dataOut = augmentPointCloudData(pc);                 % augmentation of point cloud
dataOut = formOccupancyGrid(dataOut);                % voxelization
Voxel_Data = dataOut;
end
end



function dataOut = formOccupancyGrid(data)    % voxelization
grid_idx = [32 32 32]; 
grid = pcbin(data,grid_idx);
occupancyGrid = zeros(size(grid),'single');
for ii = 1:numel(grid)
    occupancyGrid(ii) = ~isempty(grid{ii});
end

dataOut = occupancyGrid;

end

function dataOut = augmentPointCloudData(data)  % augmentation

ptCloud = pointCloud(data);


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

dataOut = ptCloud;

end
