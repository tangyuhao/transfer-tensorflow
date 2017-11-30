# Table of contents
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [Table of contents](#table-of-contents)
* [Xlearn Transfer Learning Library](#xlearn-transfer-learning-library)
	* [Setup](#setup)
	* [Usage](#usage)
		* [Quick Start](#quick-start)
	* [Methods](#methods)

<!-- /code_chunk_output -->

# Xlearn Transfer Learning Library
The Xlearn transfer learning library implements some tranfer learning or domain
adaption algorithms using TensorFlow. The script `main.py` is for both training
and evaluating the models. And it also contains some scripts for downloading and
preprocessing the data and the pretrained models.

**This library is still being heavily developed. Vast changes are expected
these days.**

## Setup
This library is Python 2&3 compatible. See `requirements.txt` for the detailed
dependencies. You may modify it if you want to use the gpu version of
TensorFlow.

## Usage
### Quick Start
Excute the following script can run Alexnet-based DAN models on the Office
dataset.

```sh
# Prepare the data
sh data/office/download.sh
# Download and process the pretrained mean
python download_mean_from_caffe.py
# Download and process the pretrained model
python download_model_from_caffe.py
# Train
python main.py
```

## Methods
**Currently it only supports DAN with Caffe alexnet base-model.**


## Google Cloud instruction
### initial setting for ubuntu 16.04
[reference link](https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0/)
```bash
#!/bin/bash
echo "Checking for CUDA and installing."
mkdir tmp
cd tmp
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda=8.0.61-1 -y
fi

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc

# install cuDNN v6.0
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

rm -rf ~/cuda
rm ${CUDNN_TAR_FILE}

sudo apt-get install python3-dev python3-pip libcupti-dev
sudo pip3 install tensorflow-gpu

```
### login to instance
1. login to our account, go to [Console](https://console.cloud.google.com/compute/instances?project=fluted-castle-186001)
2. choose 'eecs545', if it is not running, click 'start'
3. click Pull-down Menu for SSH, choose open in browser window

### gcloud
- add our account: `gcloud auth login eecs545g3@gmail.com`
- check current account: `gcloud auth list`
- set account: `gcloud config set account <account_mail>`
- copy from local: `gcloud compute scp <localfile> eecs545g3@tensorflow-train:~/download --zone us-east1-d`

### use Tmux
tmux enables you to keep your program running when you logout
Basic instruction: 
- install:
    - ubuntu: `sudo apt-get install tmux`
    - mac: `brew install tmux`
- create new session: `tmux new -s <session_name>`
- when you are in a session, all command should start with 'ctrl+b'
    - command `s`: get the list of all sessions, choose one and enter which go to that session
    - command `d`: detach from current session
    - command `:kill-session`: detach and delete current session
    - command `:kill-server`: detach and delete all sessions
- attach a session: `tmux attach-session -t <session_name>` or `tmux attach` (the first session in the list)

### GPU monitor
`nvidia-smi -l`

### Code for reference
[GOTURN-Tensorflow](https://github.com/tangyuhao/GOTURN-Tensorflow)

## To do list
- [ ] Add summary and checkpoint code
- [ ] add Adversarial code
- [ ] report