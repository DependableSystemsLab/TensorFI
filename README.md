# TensorFI: A fault injector for TensorFlow applications

TensorFI is a fault injector for TensorFlow applications written in 
Python. It instruments the Tensorflow graph to inject faults at the
level of individual operators. Unlike other fault injection tools,
the faults are injected at a higher level of abstraction, and hence
can be easily mapped to the Tensorflow graph. Further, the fault
injector can be configured though a YAML file. 

Following are the installation instructions and dependencies. For
details on how TensorFI works, how to use or modify it for your purposes,
how to contribute and licensing information, please refer our
[Wiki](https://github.com/DependableSystemsLab/TensorFI/wiki).


**Updates: 2019-07**

We now support fault injection in complex ML models such as LeNet, AlexNet, as well as support for single bit-flip injection mode. Some DNN models are provided in /Tests directory. For starters, you can try running the LeNet.py under /Tests/DNN-model/LeNet-mnist/ to inject faults in a CNN (it'll automatically download the dataset and the config file is set up).

You can now create your customized TensorFlow operations for injection, by using the built-in TensorFlow implementation, to support injection on new ML models.

Using TF Keras:

A simple MLP model implemented using TF Keras module is created and tested with TensorFI. Try it at [/Tests/keras-mnist.py](https://github.com/nniranjhana/TensorFI/blob/master/Tests/keras-mnist.py).


## 1. Supported Platforms

TensorFI has been tested on the following platforms and versions:

1. Ubuntu Linux (v 4.10) with TensorFlow (v. 1.4.1)
2. Ubuntu Linux (v 4.4) with TensorFlow (v. 1.5)  
3. Ubuntu Linux (v 16.4) with TensorFlow (v. 1.10.0) 
4. MacOSX (v10.12 and v10.13) with TensorFlow (v 1.5 and v 1.10.0) 

In general, any UNIX platform should work. We haven't tested it on Windows.

## 2. Dependencies

1. TensorFlow Framework (v 1.0 or greater)

2. Python (v2.7 or greater, but not v3.x.x)

3. PyYaml (v3 or greater)

4. SciKit module in Python

5. Sklearn module in Python

6. enum module in Python

7. numpy package (part of TensorFlow)

8. (Optional) matplotlib package in Python

9. (Optional) tkinter package in Python

## 3. Installation Instructions

### Installing as a pypi package:

We now provide TensorFI in a pypi package, so that you can install TensorFI
using pip:

   ```
   pip install TensorFI
   ```

In this way, TensorFI will be installed into the existing python environment.
Alternatively, you can install TensorFI in a virutal environment as outlined
below.

### Using the install.sh script

The easiest way to install TensorFI is to use the 
provided install.sh script which will install the Anaconda package
manager and the required dependencies, setting the 
appropriate paths.

### Manual installation

If you choose to do all of the installations yourself (if you don't want to
use the automated script or you have trouble running it), you can follow the
procedure outlined below:

1. To install, first install PyYaml v3 and above.
For example, you would type:

   ```
   pip install PyYaml
   ```

2. Install TensorFlow. You don't need to install
the GPU version if you don't want to. Make
sure you install TensorFlow for Python 2.7, not 3.
TensorFlow installation instructions can be found at:

	https://www.tensorflow.org/install/

3. Install the scipy and sklearn modules. On both
Ubutnu and MacOS, type:

   ```
   pip install scipy
   pip install sklearn
   ```

4. Set your PYTHONPATH to the TENSORFIHOME
where TENSORFIHOME is where you've installed TensorFI
(This assumes you're using Bash as your shell).

   ```
   export PYTHONPATH=$PYTHONPATH:$TENSORFIHOME
   ```

5. Run the test files by going to the TENSORFIHOME
directory and running runAll.sh in Tests. All the
tests should pass if your installation was successful. 
The script will also check if you have all of 
the above packages installed correctly.

   ```
   ./Tests/runAll.sh
   ```

   **NOTE:** The runAll script will create new subdirectories
in the TENSORFIHOME directory (faultLogs, logs and stats),
so make sure you have the permissions to do so when 
you run it (or you can manually create the directories before).
Also, make sure the python interpreter name is correct
in the script (it defaults to python) - if not, change it.

6. If you want a visual demo of TensorFI, 
  try running autoencoder.py from TensorFIHOME directory.

   ```
   python Tests/autoencoder.py
   ```

   You will see the original images (without fault injection)
and the faulty images (with fault injection) for different
fault probabilities ranging from 0.01 to 1.0 in the images.
The images are saved under the Tests/Images sub-directory 
in PNG format (make sure this directory exists first). 

   Another visual demo is when you run variational-autoencoder.py
This will also show you the original and faulty images.

   ```
   python Tests/variational-autoencoder.py
   ```

   Yet another visual demo is when you run GANs (Generative
Adversarial Networks). The images with and without faults 
are saved under the Tests/Images sub-directory.
	
   ```
   python Tests/gan.py
   ```

   **NOTE:** Both use the matplotlib and the python-tk libraries
      so you'll need to install the libraries for the demo.

If you find TensorFI useful, please cite the following paper: *"TensorFI: A configurable fault injector for TensorFlow Applications", Guanpeng Li, Karthik Pattabiraman and Nathan DeBardeleben, 8th IEEE International Workshop on Software Certification (WoSoCER), 2018.*

Find a copy of the TensorFI paper [here](http://blogs.ubc.ca/karthik/files/2018/08/TensorFI-Camera-Ready.pdf).
