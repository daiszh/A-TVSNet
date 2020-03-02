## A-TVSNet
Code for "A-TVSNet: Aggregated Two-View Stereo Network for Multi-View Stereo Depth Estimation"

## Enviroment

* Ubuntu 16.04
* Python 2.7
* Cuda 9.0 and cudnn 7.0
* Tensorflow 1.5
* other dependencies in ```requirements.txt```

## Run Examples

* Extract ```model.zip``` to ```model``` folder
* Run example
    ```bash 
    cd atvsnet
    python example.py
    ```

## Reproduce ETH3D Pointcloud Results

* Download [preprocess ETH3D dataset](https://drive.google.com/open?id=1hGft7rEFnoSrnTjY_N6vp5j1QBsGcnBB) from [MVSNet](https://github.com/YoYo000/MVSNet) and extract to ```data``` folder
* Build fusible
    ```bash 
    cd fusible
    mkdir build
    cd build
    cmake ..
    make
    ```
* Produce point cloud
    ```bash 
    cd atvsnet
    ./reproduce_pc.sh
    ```
