% clc
% clear all
% close all

 
cd('D:\Spain Data')
bag1 = rosbag('manual1b.bag'); % 
bSel_1 = select(bag1,'Topic','/icab1/velodyne_points');

Total_Messages = size(bSel_1.MessageList,1);
cd('C:\Users\hafsa\OneDrive\Desktop\Matlab Files')
xyz1=cell(1,Total_Messages);
PointCloud2_iCAB1=cell(1,Total_Messages);
% cd('C:\Users\hafsa\OneDrive\Desktop\Matlab Files\processed pointcloud\icab2')
for i=2767:1:3173
     h = figure;
     PointCloud2_iCAB1(i) = readMessages(bSel_1,i);
    ptCloud = pointCloud(readXYZ(PointCloud2_iCAB1{1,i}));
    %     data{1,i} = ptCloud;
    %     
    dataOut2 = preprocessingSteps(ptCloud.Location);
    pcshow(dataOut2)
     saveas(h,sprintf('%d.png',i));
end