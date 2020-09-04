function Data_buckets = clustering_PC(pointCloud_Processed)
% input: pre-processed point clouds
% output:  data assigned to different clusters
maxDistance = 0.3;
referenceVector = [0,0,1];
[~,inliers,outliers] = pcfitplane(pointCloud_Processed,maxDistance,referenceVector);   % fit the palne
ptCloudWithoutGround = select(pointCloud_Processed,outliers,'OutputSize','full');      % select the points without the ground plane
distThreshold = 2;                                                                     % distance threshold to cluster data 
[labels,numClusters] = pcsegdist(ptCloudWithoutGround,distThreshold);                  % cluster the data
% assign data to each cluster
Data_buckets = cell(1,numClusters);
[indxData,~,bucketNum] = find(labels);
 for i= 1:1:size(indxData,1)
       Data_buckets{1,bucketNum(i)} = [Data_buckets{1,bucketNum(i)}; ptCloudWithoutGround.Location(indxData(i),:)];
 end
 %% plot clustered pointc cloud
% numClusters = numClusters+1;
% labels(inliers) = numClusters;
% labelColorIndex = labels+1;
% pcshow(ptCloudWithoutGround.Location,labelColorIndex)
% colormap([hsv(numClusters);[0 0 0]])
% title('Point Cloud Clusters')

end