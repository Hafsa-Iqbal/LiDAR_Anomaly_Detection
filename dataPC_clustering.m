function [pcds]= dataPC_clustering(F,grid_vox) 

pcds = imageDatastore(F,'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');




function Voxel_Data  = pc_reader(filename)                   % read function
pc = table2array(readtable(filename));
 pointCloud_Processed = preprocessingSteps(pc);              % pre-processping of point clouds
 % clustering of point clouds
 Data_buckets = clustering_PC(pointCloud_Processed);
 %                                                                                                                                                                                                                                                                                                                         
% grid_vox = 32;
for clusterNumber = 1 % SHOULD BE EQUAL TO THE NUMBER OF CLUSTERS    1:1:size(Data_buckets,2)
    % FOR NOW WE ARE TAKING ONE CLUSTER AT A TIME TO SEE WHAT IS INSIDE
    % THAT CLUSTER SO YOU CAN CHANGE IT ONE BY ONE WITHOUT USING LOOP
    occupancyGrid = voxelizationConversion(pointCloud(Data_buckets{1,clusterNumber}),grid_vox);
%     VoxelsStore{1,voxIndx} = occupancyGrid;
end

Voxel_Data = occupancyGrid;
end

end
