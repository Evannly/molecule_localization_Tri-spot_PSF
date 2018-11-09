%%instruction
%setup_img_formation: this script configurates the image formation model
%and creats two data structures (img_para and sim_para) 
%necessary for generating images via img_sim_gen
%author: Hesam Mazidi

%% this section configurates the imaging model
% addpath('.\functions')
addpath(genpath(pwd))

% config_exp_img_para: produces the function get_exp_img_para to request 
%and update new imaging parameters, should it be needed


get_exp_img_para=config_exp_img_para();

% config_mask_para: produces the function config_mask_para_h to request 
%and update  new mask parameters, should it be needed

config_mask_para_h=config_mask_para();

%% this section sets up image formation model (i.e.,img_para). If you want
%to change these settings you MUST run configuration section above.

% experimental imaging parameters 
%--------------------------------
 
imgPara=get_exp_img_para(); % cashes  imaging parameters


%  phase mask mounted on the SLM and its parameters
%--------------------------------------------------

phase_mask_para=config_mask_para_h(); % cashes parameters related to the
% mask, SLM and pupil plane

Rescale_PhaseMask=config_scale_mask(phase_mask_para,imgPara);

% produces a new  function (Rescale_PhaseMask) to produce (i) a scale phase 
% mask (pmask) to match the imaging system and (ii)  a
% structure (MaskStruct) to save related parameters

[pmask,MaskStruct]=Rescale_PhaseMask(); % matches the phase mask to the
%imaging system

%% this section sets up fixed simulation parameters (i.e., sim_para). You 
% need to set sim_para.density yourself since it is not a fixed parameter

% simulation parameters
%----------------------

% random generate density range from (0.003,0.01)
density = 0.003+rand()*0.007;

disp('requesting simulation parameters ...')

prompt1={

	'image size ([width,height] in number of camera pixels)',...
	'mean brightness (number of photons) per blinking event',...
	'mean background photons per pixel',...
    'density (default: random number in the range (0.005,0.01))',...
	};

	dlgTitle='fixed simulation parameters';

	num_lines=repmat([1,100],size(prompt1,2),1);
            defaultans={'[400,400]',...
                '5000',...
                '10',...
                num2str(density),...
                };

    input=inputdlg(prompt1,dlgTitle,num_lines,defaultans);

     sim_para.fov_size = str2num(char(input(1)));
	 sim_para.brightness = str2num(char(input(2)));
	 sim_para.background = str2num(char(input(3)));
     sim_para.density = str2num(char(input(4)));
	






