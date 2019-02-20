#!/bin/bash

folder_name=./install_GPU_USRP

if [ -f /etc/os-release ]; then
    # freedesktop.org and systemd
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    # linuxbase.org
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    # For some versions of Debian/Ubuntu without lsb_release command
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    # Older Debian/Ubuntu/etc.
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
    # Older SuSE/etc.
    ...
elif [ -f /etc/redhat-release ]; then
    # Older Red Hat, CentOS, etc.
    ...
else
    # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
    OS=$(uname -s)
    VER=$(uname -r)
fi

clear
echo "Installing GPU/USRP software..."
echo "detected OS:"
echo $OS $VER
if [ $VER != 18.04 ]; then
echo "WARNING: this script has only been tested with Ubuntu 18.04"
fi

if [ "$EUID" -ne 0 ]
  then echo "Please run this script as root"
  exit
fi
#apt-get update
#sudo apt-add-repository -y ppa:graphics-drivers/ppa
apt-get update
apt-get -y install make
apt-get -y install cmake build-essential
apt-get -y install net-tools
apt-get -y install python-pip python-dev build-essential 
sudo -H pip install --upgrade pip
sudo -H pip install requests
sudo -H pip install mako
sudo apt-get -y install git
sudo apt-get -y install vim
sudo apt-get -y install htop
sudo apt-get install libusb-dev
sudo apt-get -y install libboost-all-dev
sudo apt-get -y install gcc-6.1,g++-6.1
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
sudo update-alternatives --set gcc "/usr/bin/gcc-6"
sudo update-alternatives --set g++ "/usr/bin/g++-6"
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
sudo apt-get -y install nvidia-cuda-toolkit
cd ~
mkdir $folder_name
cd $folder_name
git clone https://github.com/EttusResearch/uhd.git
cd uhd/host/
mkdir -p "build"
cd build

#cmake .. #-DENABLE_OCTOCLOCK=OFF
cmake ..

make -j12 install
ldconfig
cd ../..
git clone https://github.com/zchee/cuda-sample.git
sudo cp -r cuda-sample/common/inc* /usr/include/






#sudo apt-get -y install lightdm
#sudo apt-get -y install compiz
#sudo dpkg-reconfigure lightdm


