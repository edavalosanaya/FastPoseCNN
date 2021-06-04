# FastPoseCNN: Real-time Monocular Category-Level 6D Pose and Size Estimation Framework

Created by [Eduardo Davalos Anaya](http://edavalosanaya.com) and [Mehran Aminian](https://www.stmarytx.edu/academics/faculty/mehran-aminian/) from [St. Mary's University](https://www.stmarytx.edu).

![model_intermediate_data_representation](https://user-images.githubusercontent.com/40870026/120583636-3e531180-c3f4-11eb-91d0-dbdfa866784a.png)

Our method uses multiple representations to reconstruct an object's pose and size physical parameters. By decoupling these parameters, the framework achieves better performance and excellent inference speed.

## Introduction

![model_overall_architecture](https://user-images.githubusercontent.com/40870026/120583699-5aef4980-c3f4-11eb-9e4a-1c18c3c89ee9.png)

This PyTorch project is the implementation of my thesis, [FastPoseCNN: Real-time Monocular Category-Level 6D Pose and Size Estimation Framework](https://github.com/edavalosanaya/FastPoseCNN/files/6588916/Eduardo_s_Masters_Thesis.pdf). **Note**: that this thesis is just a proof of concept and requires more development to fully become a stable and commerically liable solution. That being said, FastPoseCNN provides an excellent tradeoff between speed, accuracy, and universaility. 

Information about the project directory and file structure.
```
FastPoseCNN
|   README.md
|
|---datasets                                # location of all datasets
|   |
|   |---NOCS                                # dataset used for most experiments
|        
|---source_code
    |   environment_linux.yaml              # dependency files (strict for linux)
    |   environment.yaml                    # relaxed dependency files
    |
    |---FastPoseCNN
        |   .env                            # environmental variables file
        |   config.py                       # Contains hyperparameter container 
        |   setup_env.py                    # Script to setup environment vars.
        |   train.py                        # Script for all training routines 
        |   evaluate.py                     # Script for all evaluation routines
        |   inference.py                    # Script for inference tests
        |   ...
        |   
        |---lib                             # Directory with all PyTorch GPU code
        |   aggregation_layer.py
        |   gpu_tensor_funcs.py
        |   ...
        |   |
        |   |---ransac_voting_gpu_layer     # PVNet's hough voting implementation
        |
        |---tools                           # Numpy+PyTorch generic tools
            create_meta+.py
            visualization.py
            ...
```
## Requirements

The specific libraries and their versions can be found in the `environment.yaml` (less strict) and `environment_linux.yaml` (more strict linux requirements). Overall, the most important dependecy requirements are the following:

+ `python==3.8.5`
+ `pytorch==1.8.0`
+ `torchvision==0.8.2`
+ `cudatoolkit==10.2`
+ `numpy==1.19.2`

Also, this project used the Hough Voting scheme and implementation from [PVNet](https://github.com/zju3dv/pvnet). The authors perform fantasic research, and without their released code, this project wouldn't be possible. Below, we have provided a brief citation to their GitHub repository and project page. 

> [PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation](https://arxiv.org/pdf/1812.11788.pdf)                     
> Sida Peng, Yuan Liu, Qixing Huang, Xiaowei Xhou, Hujun Bao                    
> CVPR 2019 oral                                                                
> [Project Page](https://zju3dv.github.io/pvnet/)

We placed their Hough Voting scheme within the `lib` directory. Intructions to compile the cuda source code is provided within PVNet GitHub's installation section. 

## Datasets

For this research, we used NOCS CAMERA and TEST datasets. **Beware**: the CAMERA dataset is very large (~140 GB). These datasets can be downloaded here: 

+ CAMERA: [training](http://download.cs.stanford.edu/orion/nocs/camera_train.zip)/[test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip)
+ REAL: [training](http://download.cs.stanford.edu/orion/nocs/real_train.zip)/[test](http://download.cs.stanford.edu/orion/nocs/real_test.zip)
+ [Object Meshes](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)

We would like to personal thank the NOCS authors for providing these datasets. Below is an another brief GitHub link:

> [Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation](https://arxiv.org/pdf/1901.02970.pdf)     
> Created by He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, Leonidas J. Guibas from Stanford University, Goodle Inc., Princeton University, Facebook AI Research.                                               
> CVPR 2019 oral                            
> [Project Page](https://geometry.stanford.edu/projects/NOCS_CVPR2019/)                                             

## Training

Before running the `train.py` script, we recommend that you modify the `HPARAM` variable that defines the overall hyperparameters used during training. More information about these hyperparameters can be found in `config.py` file. 

For training, we used the `config.MASK_TRAINING` and `config.HEAD_TRAINING` preset HPARAMs to train the model in a two stage system. Once you have modified your hyperparameters, you can run the training script with the following command:

```
python train.py 
```

Any hyperparameter can be changed in by adding `--<HPARAM NAME>=<HPARAM VALUE>`.

## Evaluation

Before evaluating, download the NOCS dataset and the weights provided in the [releases page](https://github.com/edavalosanaya/FastPoseCNN/releases).  Additionally, modify the NOCS dataset by rename four folders to match the structure shown below. This is to simply the loading of the datasets' samples.

```
NOCS
|
|---camera
|   |
|   |---train
|   |   ...
|   |   
|   |---val
|       ...
|
|---real
    |
    |---train
    |   ...
    |
    |---test
        ...
```

After making these modifications, please execute the following commands:

```
python create_meta+.py --DATASET_NAME=camera --SUBSET_DATASET_NAME=train
python create_meta+.py --DATASET_NAME=cemera --SUBSET_DATASET_NAME=val
python create_meta+.py --DATASET_NAME=real   --SUBSET_DATASET_NAME=train
python create_meta+.py --DATASET_NAME=real   --SUBSET_DATASET_NAME=test
```

After all this steps, you should be able to execute the `evalute.py` routine. Just remember to modify the CHECKPOINT hyperparameter to reflect the location of the downloaded weights.

```
python evaluate.py --CHECKPOINT=<weights path>
```

Here is an example output of the FastPoseCNN framework using the provided in this repository.

![example_output](https://user-images.githubusercontent.com/40870026/120583767-7195a080-c3f4-11eb-96da-854f5a3ae19b.png)
