function [imagey,position_para,orientation_par,signal]=img_sim_gen(img_para,sim_para,pmask)
%img_sim generates images of randomly located and oriented molecules in
%x-y according to img_para and sim_para. It uses the vectorial model and
% the theory of this function is based on  Adam S. Backer and 
% W. E. Moerner, Opt. Express 23, 4255-4276 (2015)
%->---
%input
%->---
%img_par:     	 structure
%img_para.lambda--   	 emission wavelength
%img_para.n--       	 refractive index
%img_para.Mag--          imaging system mgnification
%img_para.NA--           numerical aperture
%
%sim_para:       structure
%sim_para.fov_size--	 image size ([width,height] in number of camera pixels)
%sim_para.density--		 molecular-blinking density (number of molecules per um^2)
%sim_para.brightness--   mean brightness (number of photons) per blinking event
%sim_para.background--	 mean background photon per pixel
%
%pmask--  		 phase mask
%---->-
%output
%---->-
%position_par:    structure,
%position_par.z-- 		defocus    size (1,num_molecule) (z position)
%position_par.x-- 		x position size (1,num_molecule)
%position_par.y-- 		y position size (1,num_molecule)
%
%orientation_par:  structure containing  molecular orientation info

sampling_size=size(pmask,1);
pixel_size=img_para.pixel_size;
M=img_para.Mag; % imaging system magnification
fov_size = sim_para.fov_size;% size of the feild of veiw in units of camera pixels
fov_size_object_space=fov_size*(pixel_size/M); % field of view size in object space in (nm)
density = sim_para.density;% number of molecular parameters devided by area of the field of view
background = sim_para.background; % background photons per pixel
molecule_num = poissrnd(fov_size_object_space(1)*fov_size_object_space(2)*density*1e12)+1;
imagey = zeros([fov_size+sampling_size,molecule_num]);

disp(molecule_num);
% generate molecules at random positions
% x_pixel, y_pixel is the pixel level positions
% x_subpixel, y_subpixel is the sub-pixel level positions
%-------------------------------------------------------
signal = exprnd(ones(1,1,molecule_num)*sim_para.brightness);
% brightness of molecules according to an exponential random variable
x = rand(1,molecule_num)*fov_size_object_space(1);
y = rand(1,molecule_num)*fov_size_object_space(2);
x_pixel = floor(x./(pixel_size/M));
y_pixel = floor(y./(pixel_size/M));
x_subpixel = x - x_pixel.*pixel_size/M;
y_subpixel = y - y_pixel.*pixel_size/M;

% generate molecules at random orientations
% discribe molecular orientation with 6 parameters muxx,...,muyz
%---------------------------------------------------------------
tmp_var_1 = normrnd(zeros(1,molecule_num),ones(1,molecule_num));
tmp_var_2 = normrnd(zeros(1,molecule_num),ones(1,molecule_num));
tmp_var_3 = normrnd(zeros(1,molecule_num),ones(1,molecule_num));
tmp_var = sqrt(tmp_var_1.^2+tmp_var_2.^2+tmp_var_3.^2);
muxx = (tmp_var_1./tmp_var).^2;
muyy = (tmp_var_2./tmp_var).^2;
muzz = (tmp_var_3./tmp_var).^2;
muxy = (rand(1,molecule_num)-0.5).*min([(muxx+muyy),sqrt(muxx.*muyy)*2]);
muxz = (rand(1,molecule_num)-0.5).*min([(muxx+muzz),sqrt(muxx.*muzz)*2]);
muyz = (rand(1,molecule_num)-0.5).*min([(muyy+muzz),sqrt(muzz.*muyy)*2]);

position_para.z=zeros(1,molecule_num);
position_para.x=x_subpixel;
position_para.y=y_subpixel;



% basis images of the imaging system
%-----------------------------------

% define a handle for convenience
simDipole_novotny_v2_h=@(polar_par)simDipole_novotny_v2(polar_par,position_para,img_para,pmask);

polar_par.phiD=zeros(1,molecule_num); 
polar_par.thetaD=pi/2*ones(1,molecule_num);

XXy = simDipole_novotny_v2_h(polar_par);

polar_par.phiD=pi/2*ones(1,molecule_num); 
polar_par.thetaD=pi/2*ones(1,molecule_num);

YYy = simDipole_novotny_v2_h(polar_par);

polar_par.phiD=pi/2*ones(1,molecule_num); 
polar_par.thetaD=0*ones(1,molecule_num);

ZZy = simDipole_novotny_v2_h(polar_par);

polar_par.phiD=pi/4*ones(1,molecule_num); 
polar_par.thetaD=pi/2*ones(1,molecule_num);

XYytmp = simDipole_novotny_v2_h(polar_par);

polar_par.phiD=0*ones(1,molecule_num); 
polar_par.thetaD=pi/4*ones(1,molecule_num);

XZytmp = simDipole_novotny_v2_h(polar_par);

polar_par.phiD=pi/2*ones(1,molecule_num); 
polar_par.thetaD=pi/4*ones(1,molecule_num);

YZytmp = simDipole_novotny_v2_h(polar_par);

XYy = 2*XYytmp - XXy - YYy;
XZy = 2*XZytmp - XXy - ZZy;
YZy = 2*YZytmp - YYy - ZZy;

% image from this molecule
%-------------------------
imgy = bsxfun(@times,XXy,reshape(muxx,1,1,molecule_num))...
    + bsxfun(@times,YYy,reshape(muyy,1,1,molecule_num))+...
    + bsxfun(@times,ZZy,reshape(muzz,1,1,molecule_num)) +...
    bsxfun(@times,XYy, reshape(muxy,1,1,molecule_num))+...
    bsxfun(@times,XZy,reshape(muxz,1,1,molecule_num))+...
    bsxfun(@times,YZy,reshape(muyz,1,1,molecule_num));

% final image

for nn=1:molecule_num

imagey(y_pixel(nn)+1:y_pixel(nn)+sampling_size,x_pixel(nn)+1:x_pixel(nn)+sampling_size,nn) = ...
    imagey(y_pixel(nn)+1:y_pixel(nn)+sampling_size,x_pixel(nn)+1:x_pixel(nn)+sampling_size,nn)+imgy(:,:,nn);
end
% cropping the image to match the fov_size
imagey = imagey(sampling_size/2+1:fov_size(1)+...
    sampling_size/2,sampling_size/2+1:fov_size(2)+sampling_size/2,:);

% adding Poisson shot noise to the image
%---------------------------------------
%accounting for photon loss

polar_par.phiD=pi/2; polar_par.thetaD=pi/2;
position_para1.z=0;
position_para1.x=0;
position_para1.y=0;

brightness_scaling=simDipole_novotny_v2(polar_par,position_para1,img_para,pmask);


imagey = poissrnd(sum(bsxfun(@times,...
    bsxfun(@times,imagey,1/sum(sum(brightness_scaling))),...
    signal),3)+background);

% output
orientation_par.muxx=muxx;
orientation_par.muyy=muyy;
orientation_par.muzz=muzz;
orientation_par.muxy=muxy;
orientation_par.muxz=muxz;
orientation_par.muyz=muyz;

position_para.x=x;
position_para.y=y;