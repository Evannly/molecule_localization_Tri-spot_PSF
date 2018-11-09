% This file generate samples for training set. 
datapath = './data/';
mat_filename = 'TrainingSet_deepLoco_64';
mat_file_in = 'TrainingSet';
% load(fullfile(datapath,mat_file_in));
% read images, positions, weights
load(fullfile(datapath,mat_file_in));
% images, labels, labels_blur

% the number of training samples to create
[M,N,numImages] = size(images);

patch_size = 64;
% number of patches to extract from each image
num_patches = 200; 

% minimal number of emitters in each patch to avoid empty examples in case 
% of low-density conditions
minEmitters = 1;

% maximal number of training examples
maxExamples = 10000;

% number of training patches in total
ntrain = min(numImages*num_patches,maxExamples); 

img_size = 64;
max_mol = 256;
patches = zeros(patch_size,patch_size,ntrain);
weights = zeros(max_mol,ntrain);
positions = zeros([2,max_mol,ntrain]);
% orientations = zeros([6,max_mol,ntrain]);

% run over all images and construct the training examples
k = 1;
skip_counter = 0;
for imgNum = 1:numImages
    disp(imgNum);

    % limit maximal number of training examples to 15k
    if k > ntrain
        break;
    else
        
        % choose randomly patch centers to take as training examples
        indxy = ClearFromBoundary([M N],ceil(patch_size/2),num_patches);
        [rp,cp] = ind2sub([M,N],indxy);

        % extract examples
        image = images(:,:,imgNum);
        SpikesImage = labels(:,:,imgNum);
        for i=1:length(rp)  

            % if a patch doesn't contain enough emitters then skip it
            if nnz(SpikesImage(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2))) < minEmitters
                skip_counter = skip_counter + 1;
                continue;
            else
                patches(:,:,k) = image(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                    cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2));
                spikes = SpikesImage(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                    cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2));
                [rId, cId] = find(spikes);
                weights(1:length(rId),k) = ones([1,length(rId)]);
                positions(:,1:length(rId),k) = [cId, rId]';
%                 orientations(:,1,k) = [];
                k = k + 1;
                
%                 close all;
%                 figure('Position', [100, 100, 1400, 400]);
%                 subplot(1,3,1);
%                 imagesc(patches(:,:,k-1));
%                 subplot(1,3,2);
%                 imshow(spikes);
%                 subplot(1,3,3);
%                 plot(cId, rId,'ro');
%                 set(gca,'Ydir','reverse');
%                 axis([0 64 0 64]);
            end
        end
    end
end

% final resulting single patches dataset
images = patches(:,:,1:k-1);
positions = positions(:,:,1:k-1);
weights = weights(:,1:k-1);
% orientations = orientations(:,:,1:k-1);
save(fullfile(datapath,mat_filename),'images','weights','positions','-v7.3');

