!#/usr/bin/env bash

###
# Tested on Ubuntu 12.04, Ubuntu 16.04 and RedHat 6
###


###
# Install anaconda2
###

# Change version at https://repo.continuum.io/archive/ if you want
wget https://repo.continuum.io/archive/Anaconda2-5.2.0-Linux-x86_64.sh
bash Anaconda2-5.2.0-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda2-5.2.0-Linux-x86_64.sh

# Add anaconda binaries path in PATH environment variable
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 
echo "Type the following command now to reflect the path changes:"
echo "source ~/.bashrc"
