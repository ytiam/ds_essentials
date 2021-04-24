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
sudo yum-config-manager --enable rhui-REGION-rhel-server-extras
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


