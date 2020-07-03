## A-TVSNet
Code for "A-TVSNet: Aggregated Two-View Stereo Network for Multi-View Stereo Depth Estimation"

If you find this code useful in your research, please consider citing our [paper](https://arxiv.org/pdf/2003.00711.pdf):
```
@article{dai2020tvsnet,
  title={A-TVSNet: Aggregated Two-View Stereo Network for Multi-View Stereo Depth Estimation},
  author={Dai, Sizhang and Huang, Weibing},
  journal={arXiv preprint arXiv:2003.00711},
  year={2020}
}
```

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
