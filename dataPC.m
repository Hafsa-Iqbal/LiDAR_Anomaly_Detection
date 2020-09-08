function [pcds]= dataPC(F,grid_vox) 

pcds = imageDatastore(F,'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');




% % for i=1:1:100%length(pcds.Files)
% %     filename = sprintf('%d.txt',i);
% %     [pointCloud_Processed,Voxel_Data] = pc_reader(filename);
% %     train_pc_voxel{1,i} = Voxel_Data;
% %     if viewPC==1
% %         view(pplayer,pointCloud_Processed);
% %         pause(0.001)
% %     end
% % end

% save('Voxel_PCicab1','train_pc_voxel')

function Voxel_Data  = pc_reader(filename)                   % read function
pc = table2array(readtable(filename));
 pointCloud_Processed = preprocessingSteps(pc);
occupancyGrid = voxelizationConversion(pointCloud_Processed,grid_vox);
Voxel_Data = occupancyGrid;
end
end
