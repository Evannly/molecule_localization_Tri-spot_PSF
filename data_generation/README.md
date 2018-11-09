
# Generate Dataset

### Core function ###
1. img_sim_gen.m 
   Generate Tri-spot PSF images. Don't change it unless you know what you are doing.
2. img_sim_gen_iso.m
   Generate Gaussian PSF images. Not robustly work. If you can improve this function, welcome to commit.
3. Folder ./functions
   Set parameters for generation. Don't change it unless you know what you are doing.
4. Folder ./phasemask
   Define mask for Tri-spot PSF images.
   
### Generate data set for Network ###
1. setup_img_formation.m
   Set parameters needed for generation
   
2. dataset_gen.m
   Save generated microscopy images, ground truth of position and orientation.
   Data Folder Hierarchy
   ```
   data
   --train
     --image
     --position
     --ori
   ```
3. deepLoco_dataset_gen.m
   After step 2, create a .csv file containing microscopy images, weights, positions and orientation. Output of this step can be used in network.
   
4. deepLoco_generate_iso_64.m
   Given step 3, for each image, get a set of 64x64 image patches and corresponding positions and weights.
