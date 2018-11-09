% This file generate samples for training set. 

% the number of training samples to create
n_samples = 100;
fPath1 = './data/train/image/';
fPath2 = './data/train/position/';
fPath3 = './data/train/ori/';

% prompt1={
% 	'number of sampels to create',...
%     'file path to save molecule image',...
%     'file path to save position',...
%     'file path to save orientation',...
% 	};
% dlgTitle='Generating training set';
% 
% num_lines=repmat([1,100],size(prompt1,2),1);
% defaultans={num2str(n_samples),...
%     fPath1,...
%     fPath2,...
%     fPath3
%     };
% 
% input=inputdlg(prompt1,dlgTitle,num_lines,defaultans);
% 
% n_samples = str2num(char(input(1)));
% fPath1 = char(input(2));
% fPath2 = char(input(3));
% fPath3 = char(input(4));

    
% check file path for train samples
if exist(fPath1,'dir') ~= 7
    mkdir(fPath1)
end
if exist(fPath2,'dir') ~= 7
    mkdir(fPath2)
end
if exist(fPath3,'dir') ~= 7
    mkdir(fPath3)
end
    
% number training samples
D = dir(fPath1);
[r,c] = size(D);
if D(3).name == 'img1.csv'
    r = r-2;
else
    r = r-3;
end
disp(r);

for i = 1+r:n_samples+r
    [imagey,position_para,orientation_par,signal]=img_sim_gen(imgPara,sim_para,pmask);
    
    disp(i);
    
    filename1 = [fPath1,'img',num2str(i),'.csv'];
    csvwrite(filename1,imagey);
    
    position = cell2mat( struct2cell( position_para ));
    position = position./(imgPara.pixel_size/imgPara.Mag);
    filename2 = [fPath2,'position',num2str(i),'.csv'];
    csvwrite(filename2,position);
    
    orientation = cell2mat( struct2cell( orientation_par) );
    filename3 = [fPath3,'orientation',num2str(i),'.csv'];
    csvwrite(filename3,orientation);
    
end


