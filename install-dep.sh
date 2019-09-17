!#/usr/bin/env bash

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
echo "To set python path for TensorFI in anaconda, type:"
echo "ln -s <FullPathToTensorFIProject>/TensorFI/ ~/anaconda/envs/tensorfi/lib/python2.7/site-packages/"

# Activate and deactivate virtualenv

echo "To activate the 'tensorfi' anaconda virtual environment, type the following command:"
echo "source ~/anaconda/bin/activate tensorfi"

echo "You can deactivate the virtualenv after use by typing:"
echo "conda deactivate"
