# MonoASRH: Efficient Feature Aggregation and Scale-Aware Regression for Monocular 3D Object Detection

This repository hosts the official implementation of [Efficient Feature Aggregation and Scale-Aware Regression 
for Monocular 3D Object Detection](https://arxiv.org/abs/2411.02747).

<h5 align="center">

</h5>

The official results in the paper on KITTI Val Set:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoASRH</td>
        <td div align="center">28.35%</td> 
        <td div align="center">20.75%</td> 
        <td div align="center">17.56%</td> 
    </tr>  
</table>

This repo results on KITTI Val Set:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
        <td rowspan="2",div align="center">Checkpoint</td>
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoASRH</td>
        <td div align="center">28.15%</td> 
        <td div align="center">20.87%</td> 
        <td div align="center">17.61%</td> 
        <td div align="center">Comming Soon</td>
    </tr>  
    <tr>
        <td div align="center">28.38%</td> 
        <td div align="center">21.07%</td> 
        <td div align="center">17.78%</td> 
        <td div align="center">Comming Soon</td>
    </tr>
    <tr>
        <td div align="center">28.28%</td> 
        <td div align="center">21.11%</td> 
        <td div align="center">17.84%</td> 
        <td div align="center">Comming Soon</td>
    </tr>
</table>

The official results in the paper on KITTI Test Set:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Test, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoASRH</td>
        <td div align="center">26.18%</td> 
        <td div align="center">19.17%</td> 
        <td div align="center">16.92%</td> 
    </tr>  
</table>

## Installation
1. Clone this project and create a conda environment:
    ```bash
    git clone https://github.com/WYFDUT/MonoASRH.git
    cd MonoASRH

    conda create -n monoasrh python=3.9
    conda activate monoasrh
    ```
    
2. Install pytorch and torchvision matching your CUDA version:
    ```bash
    # For example, We adopt torch 1.11.0+cu113
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
    
3. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
 
4. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```bash
    │MonoASRH/
    ├──...
    │data/kitti/
    ├──ImageSets/
    ├──training/
    │   ├──image_2
    │   ├──label_2
    │   ├──calib
    ├──testing/
    │   ├──image_2
    │   ├──calib
    ```
    You can also change the data path at "dataset/root_dir" in `lib/kitti.yaml`.

## Get Started

### Train
You can modify the settings of models and training in `lib/kitti.yaml`:
  ```bash
  python tools/train_val.py
  ```
### Eval
  ```bash
  python tools/train_val.py -e
  ```
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/resume_model" in `lib/kitti.yaml`:
  ```bash
  python tools/train_val.py -t
  ```

## Acknowlegment
This repo benefits from the excellent work [GUPNet](https://github.com/SuperMHP/GUPNet/tree/main) and [MonoLSS](https://github.com/Traffic-X/MonoLSS)
