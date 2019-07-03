# Different DNN models for performing fault injection with TensorFI

- This directory contains 6 different DNN models, e.g., steering model in self-driving cars.
- The benchmarks and datasets are as follows (link to download the dataset): 
    - LeNet-4 - Mnist dataset (http://yann.lecun.com/exdb/mnist/)
    - AlexNet - Cifar-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
    - VGG16 - ImageNet (http://image-net.org/download)
    - VGG11 - German traffic sign (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
    - Comma.ai's steering model - real-world driving frame (https://github.com/SullyChen/driving-datasets)
    - Nvidia Dave steering model - real-world driving frame (https://github.com/SullyChen/driving-datasets) 
- Note that only vgg16 model is provided with a pre-trained weight (http://www.cs.toronto.edu/~frossard/post/vgg16/); so for the rest of models, you need to train the model first, before performing fault injection.
- To perform fault injection on your own model, please follow the instruction in the HOWTORUN manual.