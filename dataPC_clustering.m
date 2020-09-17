function [pcds]= dataPC_clustering(F,grid_vox)

pcds = imageDatastore(F,'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% read function
    function Voxel_Data  = pc_reader(filename)
        pc = table2array(readtable(filename));
        % pre-processping of point clouds
        pointCloud_Processed = preprocessingSteps(pc);
        % clustering of point clouds
        NearestCluster = clustering_PC(pointCloud_Processed);
        % voxelization of nearest cluster
        occupancyGrid = voxelizationConversion(pointCloud(NearestCluster),grid_vox);
        Voxel_Data = occupancyGrid;
    end

end
