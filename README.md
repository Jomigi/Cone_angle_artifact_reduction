# Cone Angle Artifact Reduction using symmetry-aware deep learning

This repository contains code that was used to perform the experiments described in the paper "Efficient High Cone-Angle Artifact Reduction in Circular Cone-Beam CT using Symmetry-Aware Deep Learning with Dimension Reduction" by Minnema et al.

## Getting started
We recommend installing Conda or Miniconda for Python3 to set up the required packages to run this code. A virtual environment in (Mini)conda can be created and activated with:

``` 
env_name=*my_name*
conda create --name $env_name python=3.6
conda activate env_name
```

To get the source code and install the required pacakges

```
git clone https://github.com/Jomigi/Cone_angle_artifact_reduction.git
cd Cone_angle_artifact_reduction
python setup.py
```

## Code description
The code in this repository is split into three parts: CTreconstruction, CNN, and Radial2Cartesian. 

### CT reconstuction
Here you can find all code that is necessary to reproduce the reconstruction of cone-beam CT scans as performed in the paper. This includes the generation of the input scans using FDK, and the generation of the ground truth using Nestorov gradient descent, and the  FDK-to-Radial interpolation step. 

To run this code, specify the datapaths in the respective files and run: 

```
python XXX # To create input
python XXX # To create ground truth
python XXX # To perform the FDK-to-Radial interpolation
```

### CNN
This includes all code necessary to train an MS-D Net or a U-Net to reduce cone-angle artifacts in cone-beam CT scans. To train the networks, specify the datapaths in train.py and then run:

```
python train.py 
```

This folder also contains the validation scheme (validation.sh) that was used to optimize the depth and dilations of MS-D Net as well as the number of epochs to train both CNNs. 

### Radial2Cartesian
This folder contain a single script which was used to performed to Radial-to-Cartesian re-sampling step. 

To run this code:
```
python radial2cartesian.py
```

## License
The code is licensed under the XXX license - see the LICENSE.md file for details. 
