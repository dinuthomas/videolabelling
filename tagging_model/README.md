## Steps to run the tagging model

### Note, this code has been tested on devices with NVIDIA V100 and A5000 GPUs. It is based on docker, therefore, the cuda and driver version of host server will does not affect its running. (Various host's cuda and driver version have also been tested.)

1.  Pull the following docker image from docker hub, which has python2.x and tensorflow-1.8.0-gpu, using the cmd:

    ```bash
    docker pull tensorflow/tensorflow:1.8.0-devel-gpu
    ```
2.  Download the pre-trained [model params](https://drive.google.com/file/d/1JaR6oPR1v4l-O8lnHIpFxyfNnUSGmd8Y/view?usp=sharing) from google drive and decompress it in the current folder using the cmd:
    ```
    tar -zxvf ckpt.tgz
    ```
    
3.  Start the container (if you do not have nvidia-docker, pls install it (on Ubuntu) by: apt intall nvidia-docker2).

    ```bash
    nvidia-docker run -it --rm -v `pwd`:/tagging_model 5f07c3dd29cc /bin/bash
    ```

4.  Use eval.sh to run the inference code, will output top 20 predictions of the input video. If you run for the first time, it will take serveral minutes. Specify the used GPU id, the input video feature path, the pre-trained model path in the eval.sh.

    ```bash
    cd /tagging_model && sh infer.sh
    ```
