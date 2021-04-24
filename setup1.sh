#!/bin/bash
# INSTALL GIT & WGET #
sudo yum update -y
sudo yum install -y wget
sudo yum install -y git
# DO SYSTEM UPDATE #
# INSTALL ORACLE JDK #
wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" https://download.oracle.com/otn-pub/java/jdk/13.0.2+8/d4173c853231432d94f001e99d882ca7/jdk-13.0.2_linux-x64_bin.rpm
sudo rpm -ivh jdk-13.0.2_linux-x64_bin.rpm
rm jdk-13.0.2_linux-x64_bin.rpm
# INSTALL DOCKER #
sudo yum install -y yum-utils
sudo yum-config-manager --enable rhel-7-server-rhui-extras-rpms
sudo yum install -y docker
# POSTGRESQL INSTALLATION #
sudo yum install -y postgresql postgresql-server postgresql-devel postgresql-contrib postgresql-docs
sudo postgresql-setup initdb
sudo service postgresql start
# AWS CLI INSTALLATION #
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
sudo python get-pip.py
sudo pip install awscli
rm get-pip.py
mkdir .aws
echo "[default]">.aws/config
read -p "AWS Region : "
echo "region = $REPLY">>.aws/config
echo "signature_version = s3v4">>.aws/config
echo "[default]">.aws/credentials
read -p "AWS Access Key ID : "
echo "aws_access_key_id = $REPLY">>.aws/credentials
read -p "AWS Secret Access Key : "
echo "aws_secret_access_key = $REPLY">>.aws/credentials
# NodeJS Installation #
sudo yum install -y gcc gcc-c++ kernel-devel make poppler-utils pkgconfig poppler-cpp-devel python-devel redhat-rpm-config GraphicsMagick-c++-devel boost-devel 

curl -sL https://rpm.nodesource.com/setup_12.x | sudo -E bash -
sudo yum install -y yarn
sudo yum install -y nodejs
# Enabling Epel Repo #
wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum install -y epel-release-latest-7.noarch.rpm
rm epel-release-latest-7.noarch.rpm
# ANACONDA INSTALLATION #
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh


sudo yum groupinstall -y ‘Development Tools’
sudo yum install -y libcurl-devel openssl-devel libxml2-devel
sudo yum --enablerepo=rhel-7-server-rhui-optional-rpms install -y R





# For GPU
sudo yum install kernel-devel-3.10.0-862.11.6.el7.x86_64 kernel-headers-3.10.0-862.11.6.el7.x86_64

sudo yum-config-manager --enable rhel-7-server-rhui-optional-rpms
sudo yum-config-manager --enable rhel-7-server-rhui-extras-rpms
sudo yum -y update
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-rhel7-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm
sudo yum clean all

sudo yum -y install nvidia-driver-latest-dkms cuda
sudo yum -y install cuda-drivers



pip install cupy-cuda102
pip install cupy

conda install tensorflow-gpu 
conda install keras-gpu  
conda install -c anaconda cudatoolkit

conda install pytorch torchvision -c pytorch

pip install dask-cudf
conda install pytest
pip install "dask[complete]"
pip install pycuda

conda install chainer
conda install caffe-gpu
conda install py-xgboost-gpu
conda install mxnet-gpu
conda install accelerate

pip install cuml
pip install cudf


