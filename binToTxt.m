% for processing sydney data
% point clouds data store in a .bin format
% this code use to read .bin file and store it into txt format
% this code is write to read and write one file at a time

clear all

% read a bin file 
% give location to the file where your data is store 
fname = 'C:\Users\hafsa\OneDrive\Desktop\sydneyThreeClasses\temp\pedestrain\pedestrian.28.6994.bin';
fid = fopen(fname, 'r');
c = onCleanup(@() fclose(fid));

fseek(fid,10,-1); % Move to the first X point location 10 bytes from beginning
X = fread(fid,inf,'single',30);
fseek(fid,14,-1);
Y = fread(fid,inf,'single',30);
fseek(fid,18,-1);
Z = fread(fid,inf,'single',30);

fseek(fid,8,-1);
intensity = fread(fid,inf,'uint8',33);

pointData = [X,Y,Z];
% store point data in a txt file 
txtFile_name = 1;
FileName = sprintf('%d.txt',txtFile_name);  % increase or reduce the number of zeros befor %d
file = fopen(FileName,'wt');
fprintf(file,'%f %f %f\n',pointData);
fclose(file);



