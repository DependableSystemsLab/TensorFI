This directory contains different ML models that classify MNIST dataset.
The performance of each of these models in the presence of faults is evaluated and compared. Performance is accuracy drop over different FI (Fault Injection) configuration settings.

To disable faults, set *disableInjections* to `True`. To play with other types of FI, make changes to the corresponging YAML file.

The description of each model follows:

**nn.py**	- A basic neural network architecture with 4 hidder layers.
**cnn.py**	- A basic convolutional neural network architecture.

Update README here as more models are benchmarked.
