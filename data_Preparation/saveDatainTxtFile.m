% clc
% clear all
% close all

cd('D:\Spain Data')
bag1 = rosbag('scenario3amode1.bag') ; % 
bSel_1 = select(bag1,'Topic','/icab1/velodyne_points');

Total_Messages = size(bSel_1.MessageList,1);
cd('C:\Users\hafsa\OneDrive\Desktop\Matlab Files')
xyz1=cell(1,Total_Messages);
PointCloud2_iCAB1=cell(1,Total_Messages);
for i=3041:1:3210
    PointCloud2_iCAB1(i) = readMessages(bSel_1, i);
    ptCloud = pointCloud(readXYZ(PointCloud2_iCAB1{1, i}));
    
    FileName = sprintf('%d.txt',i);  % increase or reduce the number of zeros befor %d
    file = fopen(FileName,'wt');
    fprintf(file,'%f %f %f\n',ptCloud.Location');
    fclose(file);
    
end


%% read txt files
% 
% C:\Users\hafsa\OneDrive\Desktop\Matlab Files\processed pointcloud\scenario1\manual1b_icab1
%  for i=1:1:3 % number of data files
%     filename = sprintf('%d.txt',i);
%     data = importdata(filename);
%     StoreData{1,i} = data;
%  end