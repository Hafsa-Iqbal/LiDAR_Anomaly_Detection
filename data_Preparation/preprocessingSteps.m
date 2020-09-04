function dataOut2 = preprocessingSteps(pc)
maxLidarRange = 20;
% maxLidarRangeX = 3;
% maxLidarRangeY = 10;
referenceVector = [0, 0, 1];
maxDistance = 0.5;
maxAngularDistance = 15;
annularRegionLimits = [-0.75,0.75];


%% Point Cloud Filtering
% Point cloud filtering is done to extract the region of interest from the acquired scan. Here, the region of interest is the annular region with ground and ceiling removed.
ind = (-maxLidarRange < pc(:,1) & pc(:,1) < maxLidarRange ...
    & -maxLidarRange  < pc(:,2) & pc(:,2) < maxLidarRange ...
    & (abs(pc(:,2))>abs(0.5*pc(:,1)) | pc(:,1)>0));

pcl = pointCloud(pc(ind,:));
% remove noise from data
pcl = pcdenoise(pcl);
% % Remove points on the ground plane.
[~, ~, outliers] = ...
    pcfitplane(pcl, maxDistance,referenceVector,maxAngularDistance);
pcl = select(pcl,outliers,'OutputSize','full');
%% optional steps
% % % Select points in annular region.
% ind = (pcl_wogrd.Location(:,3)<annularRegionLimits(2))&(pcl_wogrd.Location(:,3)>annularRegionLimits(1));
% pcl = select(pcl_wogrd,ind,'OutputSize','full');

%pcdownsample(pcl_wogrd,'random',randomSampleRatio);


dataOut2 =  pcl;
end