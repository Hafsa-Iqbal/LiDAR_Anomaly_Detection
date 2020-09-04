clear all
% one fila at a time
cd('C:\Users\hafsa\OneDrive\Desktop\Matlab Files\processed pointcloud\scenario1\manual1b_icab1\txtFiles')
% load point cloud you want to cluster
data = load('0010.txt');


pointCloud_Processed = preprocessingSteps(data);
% clustering 
Data_buckets = clustering_PC(pointCloud_Processed);

% path to store clustered txt files
cd('C:\Users\hafsa\OneDrive\Desktop\clusteredData\1');
% store cluster data in seperate txt 
 for i =1:1:size(Data_buckets,2)
     FileName = sprintf('%d.txt',i);  % increase or reduce the number of zeros befor %d
    file = fopen(FileName,'wt');
    DATAtemp = pointCloud(Data_buckets{1,i});
    fprintf(file,'%f %f %f\n',single(DATAtemp.Location'));
    fclose(file);
    pcshow(Data_buckets{1,5})
 end
 
 

