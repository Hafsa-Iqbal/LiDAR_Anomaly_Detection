function occupancyGrid = voxelizationConversion(PointCloud_processed,grid_i)

ptCloud = PointCloud_processed;
% % Spatially bin the point cloud into a 32-by-32-by-32 grid.
indices_occupancy = pcbin(ptCloud,[grid_i grid_i grid_i]);
occupancyGrid = cellfun(@(c) ~isempty(c),indices_occupancy);

end
%%
% for i = 1:length(ptCloud2)
%     ptCloud = ptCloud2{1,i};
%     % % Spatially bin the point cloud into a 32-by-32-by-1 grid.
%     indices_occupancy = pcbin(ptCloud,[32 32 1]);
%     occupancyGrid = cellfun(@(c) ~isempty(c),indices_occupancy);
%     % % Build a density grid.
%     densityGrid = cellfun(@(c) ~isempty(c),indices_density);
%
%     % % Display the density grid.
%     figure;
%     imagesc(densityGrid);
%
% end