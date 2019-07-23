   ***The following documentation details how to run the experiments and reproduce the results in the TensorFI paper***

**0. How to run the experiments**

1. Two options to install TensorFI module and set up the environment: 

   - Install the TensorFI on the *local* python environment ==> Follow in instructions on README.md for installation; 
   - Install the TensorFI on the *virtual* python environment in Anaconda ==> Run the *Install.sh*, which automatically setup the TensorFI module in Anaconda and instruct how to setup python path for TensorFI.

2. Set you current path under the TensorFI project (i.e., /path/to/TensorFI-master/TensorFI).

3. To a run specific test (e.g., Tests/linear_regression.py), you would type:

   ```
   python Tests/linear_regression.py
   ```

4. To run multiple tests, you would type:

   ```
   Tests/runAll.sh
   ```

   you can also specify which tests to run by modifying the *runAll.sh* file.

**1. How to reproduce the results in the TensorFI paper** 

1. To run all the tests for multiple times, you would type:

   ```
   ./runAllExperimentalTest.sh
   ```

   The results (i.e., the accuracy drop after fault injection) are stored under /experimentalTest/accuracyResults/ *separately* for each test (so make sure you have a subfolder named "accuracyResults" under /experimentalTest). You can compare the average accuracy drop in each test for different dataset under /experimentalTest/accuracyResults/ with those in the paper.

2. To run a specific test for multiple times, e.g., to run  *adult_logistic_regression.py* for 100 times, you would type:

   ```
   python experimentalTest/run_multiple.py 100  experimentalTest/adult_logistic_regression.py  experimentalTest/accuracyResults/adult_LR.csv
   ```

   There are 3 args to specify: 1) the number of runs; 2) the path for the tests (i.e., the .py file); 3) the path for the file to log the result of accuracy drop.

3. To run a specific test for one time (e.g., to run *adult_logistic_regression.py*), you would type:

   ``` 
   python experimentalTest/adult_logistic_regression.py experimental/accuracyResults/adult_LR.csv
   ```

   There is 1 arg to specify: 1) the path for the file to log the result of accuracy drop.
   
   
***Below is a general approach on how to run the experiment on a ML program after the successful installation of TensorFI***

   
1. Insert the following statement AFTER the training phase and BEFORE the inference phase in your main ML program (TensorFI is intended for fault injection in the inference phase). See the test examples in /experimentalTest for more details.

   ```
   fi = ti.TensorFI(tf.Session(), logLevel = 50, name = "convolutional", disableInjections=False)
   ```
   
   You can configure the parameters accordingly.

2. Follow the instruction to configure the configuration file. By default, it's in the /confFiles/default.yaml file (including the instructions).

3. Run the program and you can observe the output of the ML model under fault.



**Questions ? Contact Karthik Pattabiraman (karthikp@ece.ubc.ca)**
