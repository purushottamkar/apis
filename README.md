# APIS: Alternating Projection onto Incoherent Sub-spaces.
This repository presents an implementation of the APIS algorithm. The accompanying paper can be accessed at [https://doi.org/10.1007/s10994-021-06045-z]


## Setup
The python packages required to run the code are listed in requirement.txt, with the respective version number at the time of publishing the code. You may use the following command to install the libraries with the given versions if you are a pip user. 

  pip3 install -r requirements.txt
  
To install the latest version, please drop the version numbers from the requirements.txt file.

## Dataset
Most experiments are performed on synthetic data, which are generated on the fly. For image experiments, we have used the standard Set12 dataset with images scaled to 512 * 512. It is available in the img/input/Set12 folder.

## Executing APIS
The application of APIS is demonstrated in three areas:

1. **Robust non-parametric Kernel Regression:**
The *ker_reg* folder contains four jupyter notebook files applying APIS for functions shown in Fig. 3 of the paper.

2. **Robust Linear Regression:**
The *lin_reg* folder contains two jupyter notebook files applying APIS for robust linear regression setting in Fig. 4.

3. **Image Denoising:** 
In the img folder, the *Adversarial_Salt_Pepper_Corruption.ipynb* file gives APIS for adversarial salt-pepper corruption as reported in Fig. 6 and *Block_Corruption.ipynb* gives APIS for block corruption as reported in Fig. 7. Additionally, the *Generate_Corrupted_Image.ipynb* file gives code for introducing these corruption.

## Expected Duration
Following are the time taken to run the respective notebooks on a 64-bit machine with Intel® CoreTM i7-6500U CPU @ 2.50 GHz, 4 cores, 16 GB RAM and Ubuntu 16.04 OS.

kerreg_sin_x.ipynb    12.51 sec

kerreg_x_sinx.ipynb   13.12 sec

kerreg_x.ipynb        11.78 sec

kerreg_poly_x.ipynb   11.75 sec

LinearReg_with_Gaussian_noise.ipynb       4.21 sec

LinearReg_without_Gaussian_noise.ipynb    4.10 sec

Adversarial_Salt_Pepper_Corruption.ipynb  1.90 sec (per image)

Block_Corruption.ipynb                     83 sec (per image)

## Contributing
This repository is released under the MIT license. If you would like to submit a bugfix or an enhancement to APIS, please open an issue on this GitHub repository. We welcome other suggestions and comments too (please mail the corresponding author at purushot@cse.iitk.ac.in)

## License
This repository is licensed under the MIT license - please see the [LICENSE](LICENSE) file for details.

## Reference
Bhaskar Mukhoty, Subhajit Dutta, and Purushottam Kar. Robust Non-parametric Regression via Incoherent Subspace Projections. Machine Learning (2021) (available at https://doi.org/10.1007/s10994-021-06045-z)
