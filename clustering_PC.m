function NearestCluster = clustering_PC(pointCloud_Processed)
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

MeanClusters = [];
for i= 1:1:numClusters
    if size(Data_buckets{1,i},1)==1
        Data_buckets{1,i} = [];                        % empty the bucket having only one point
    else
        MeanClusters = [MeanClusters ; mean(Data_buckets{1,i})];
        
    end
end
emptyCells = cellfun('isempty', Data_buckets); % select the buckets which are empty
Data_buckets(emptyCells) = [];                 % remove the empty buckets
% distance between mean and reference vector and select the min distance
[meanNearestCluster, numNearestCluster ]= min(pdist2(MeanClusters,referenceVector));
% select corresponding/nearest cluster 
NearestCluster = Data_buckets{1,numNearestCluster};

%% plot clustered pointc cloud
% numClusters = numClusters+1;
% labels(inliers) = numClusters;
% labelColorIndex = labels+1;
% pcshow(ptCloudWithoutGround.Location,labelColorIndex)
% colormap([hsv(numClusters);[0 0 0]])
% title('Point Cloud Clusters')

end