function get_mask_para=config_mask_para()
% Generates a function, Getpara to request for phase mask parameters
% and cashes the parameters
%author: Hesam Mazidi

configed_already=0;
cashed_mask_para=[];

    function mask_para=GetPara()
        if (~configed_already || isempty(cashed_mask_para))
            display('requesting experimental mask parameters ...')
            prompt2={'Mask name (without .bmp)',...
                'Radius of the pupil',... % in the units of SLM pixels
                'x shift for the center position of the new mask',...
                'y shift for the center position of the new mask'...
                'Mask rotation (#*90 degree)',...
                'FileFormat(.bmp or .mat)', ...
                'Isolation distance'
                };
            dlgTitle='experimental parameters of phase mask';
            num_lines=repmat([1,80],size(prompt2,2),1);
            defaultans={
                'tri-spot',...
                '80',...
                '0',...
                '0',...
                '0',...
                '.bmp',...
                '90'
                };
            
            configed_already=1;
            cashed_mask_para=inputdlg(prompt2,dlgTitle,num_lines,defaultans);
            
        else
            display('already have mask parameters ...')
            
        end
        
        mask_para=cashed_mask_para;
    end

get_mask_para=@GetPara;
end
