% This file generate samples for training set. 
datapath = './data/';
mat_filename = 'TrainingSet_deepLoco'; 

% the number of training samples to create
n_samples = 800;
fPath1 = './data/train/image/';
fPath2 = './data/train/position/';
fPath3 = './data/train/ori/';
    
% number training samples
D = dir(fPath1);
[r,c] = size(D);
if D(3).name == 'img1.csv'
    s = 2;
else
    s = 3;
end
disp(s);

img_size = 400;
max_mol = 256;
images = zeros([img_size,img_size,n_samples]);
weights = zeros(max_mol,n_samples);
positions = zeros([2,max_mol,n_samples]);
orientations = zeros([6,max_mol,n_samples]);
%position_par.z-- 		defocus    size (1,num_molecule) (z position)
%position_par.x-- 		x position size (1,num_molecule)
%position_par.y-- 		y position size (1,num_molecule)
for i = 1:r-s
    filename1 = [fPath1,'img',num2str(i),'.csv'];
    img = csvread(filename1);
    if size(img) == 401
        img = img(1:400,1:400);
    end
    
    filename2 = [fPath2,'position',num2str(i),'.csv'];
    position = csvread(filename2);
    
    
%     orientation = cell2mat( struct2cell( orientation_par) );
    filename3 = [fPath3,'orientation',num2str(i),'.csv'];
    orientation = csvread(filename3);
    
    images(:,:,i) = img;
    shape = size(position);
    weights(1:shape(2),i) = ones([1,shape(2)]);
    positions(:,1:shape(2),i) = position(2:3,:);
    orientations(:,1:shape(2),i) = orientation;
end

save(fullfile(datapath,mat_filename),'images','weights','positions','orientations','-v7.3');

