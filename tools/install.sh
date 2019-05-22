#!/bin/sh

folder_name=./src_usrpgpu

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

before_reboot(){
    clear
    cd ~
    mkdir $folder_name
    cd $folder_name
    echo "Installing GPU/USRP software (part one)..."
    echo "detected OS:"
    echo $OS $VER
    if [ $VER != 18.04 ]; then
    echo "WARNING: this script has only been tested with Ubuntu 18.04"
    fi

    if [ "$EUID" -ne 0 ]
      then echo "Please run this script as root"
      exit
    fi

    # add needed repos
    sudo add-apt-repository ppa:mhier/libboost-latest -y


    apt-get update
    apt-get -y install make cmake build-essential
    apt-get -y install net-tools python-pip python-dev build-essential
    sudo -H pip install --upgrade pip
    sudo -H pip install requests mako numpy

    #utilities
    sudo apt-get -y install git vim htop libusb-dev

    #install boost
    sudo apt-get -y install libboost

    # after cuda 10.1 gcc6 is not needed
    #sudo apt-get -y install gcc-6.1,g++-6.1
    #sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
    #sudo update-alternatives --set gcc "/usr/bin/gcc-6"
    #sudo update-alternatives --set g++ "/usr/bin/g++-6"
    #sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
    #sudo apt-get -y install nvidia-cuda-toolkit

    #install cuda
    sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev -y
    echo "Downloading CUDA..."
    wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
    echo "Installing CUDA toolkit and samples..."
    sudo sh cuda_10.1.168_418.67_linux.run --silent --toolkit --samples
    echo "Banning nouveau driver..."
    sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    echo "Adding tools to path..."
    sudo echo 'PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}' >> ~/.profile
    sudo echo 'LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile

    printf 'Reboot needed. After reboot, change tty terminal, (ctrl+alt+F2) and relaunch this script to complete installation process. press [ENTER] to reboot...'
    read _

}


after_reboot(){
    echo "Installing GPU/USRP software (part two)..."
    cd ~
    cd $folder_name
    echo "Disabling services..."
    sudo service gdm stop
    sudo service lightdm stop
    echo "installing cuda driver..."
    sudo sh cuda_10.1.168_418.67_linux.run --silent --driver
    echo "Installing UHD..."
    git clone https://github.com/EttusResearch/uhd.git
    cd uhd/host/
    mkdir -p "build"
    cd build

    # disabling the octoclock control is needed to compile uhd 3.13
    cmake .. -DENABLE_OCTOCLOCK=OFF

    make -j12
    sudo make install
    sudo ldconfig
    printf 'DONE! Press [ENTER] to reboot...'
    read _

}


terminal_mode="$(ps hotty $$ | cut -c1-3)"
if [ "$terminal_mode" != "tty" ]; then
  before_reboot
  echo "Part one"
  sudo touch /var/run/rebooting-for-updates
  sudo update-rc.d rebooting-for-updates defaults #> /dev/null 2>&1
  sudo reboot
else
  after_reboot
  echo "Part two"
  sudo rm /var/run/rebooting-for-updates
  sudo update-rc.d rebooting-for-updates remove  #> /dev/null 2>&1
  sudo reboot
fi

#terminal_mode=""
#if [ -f /var/run/rebooting-for-updates ]; then
#    terminal_mode="$(ps hotty $$ | cut -c1-3)"
#    if [ "$terminal_mode" != "tty" ]; then
#        echo "Not running in tty terminal, please press ctrl+alt+F2, login and rerun'"
#    else
#      #after_reboot
#      echo "part two"
#      sudo rm /var/run/rebooting-for-updates
#      sudo update-rc.d rebooting-for-updates remove  #> /dev/null 2>&1
#      #sudo reboot
#    fi
#else
#    before_reboot
#    echo "Part one"
#    sudo touch /var/run/rebooting-for-updates
#    sudo update-rc.d rebooting-for-updates defaults #> /dev/null 2>&1
#    sudo reboot
#fi
