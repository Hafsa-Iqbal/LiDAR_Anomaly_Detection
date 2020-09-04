function [pcds]= dataPC_clustering(F,grid_vox)

pcds = imageDatastore(F,'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');




    function Voxel_Data  = pc_reader(filename)                   % read function
        pc = table2array(readtable(filename));
        occupancyGrid = voxelizationConversion(pointCloud(pc),grid_vox);
        Voxel_Data = occupancyGrid;
    end

end
