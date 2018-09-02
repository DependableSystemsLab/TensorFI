###
# Tested on Ubuntu 12.04, Ubuntu 16.04 and RedHat 6
###


###
# Install anaconda3
###

# Change version at https://repo.continuum.io/archive/ if you want
wget https://repo.continuum.io/archive/Anaconda2-5.2.0-Linux-x86_64.sh
bash Anaconda2-5.2.0-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda2-5.2.0-Linux-x86_64.sh

# Add anaconda binaries path in PATH environment variable
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc

# Update the package
conda update conda


###
# Create an anaconda3 environment called "tensorfi" and install dependencies
###
conda create -n tensorfi python=2.7 anaconda
conda install -n tensorfi scikit-learn   
conda install -n tensorfi yaml
conda install -n tensorfi tensorflow


###
# Set python path for TensorFI
###
echo "To set python path for TensorFI in anaconda, type :"
echo "ln -s FullPathToTensorFiProject/TensorFI/ ~/anaconda/envs/tensorfi/lib/python2.7/site-packages/"
echo "To activate tensorfi anaconda environment, type :"
echo "source ~/anaconda/bin/activate tensorfi"
