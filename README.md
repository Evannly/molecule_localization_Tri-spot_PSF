# Molecule Localization

### Data Generation ###
See README file in ./data_generation for generation guide.

Here is a example dataset of Gaussian PSF images with size 64x64 (./data_generation/TrainingSet_deepLoco_iso_64.mat.zip). Ready to be used in network.
### Using deepLoco for localization ###
Current network is for 64x64 images.
1. Run train.py for training.
2. Run test.py for testing.

#### Example ####
![Left: Gaussian PSF image; middle: true location; right: prediction with confidence](https://github.com/Evannly/molecule_localization_Tri-spot_PSF/blob/master/doc/Capture2.PNG)

### Reference ###
[DeepLoco: Fast 3D Localization Microscopy Using Neural Networks](https://www.biorxiv.org/content/biorxiv/early/2018/02/16/267096.full.pdf)
