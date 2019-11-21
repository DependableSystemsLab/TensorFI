This directory contains different convolutional neural network (CNN) models.

The performance of each of these models in the presence of faults is evaluated and compared. Performance is measured as:
- accuracy drop over different FI (Fault Injection) configuration settings.
- SDC rate (silent data corruption): over given inputs, probability of corruption in their outputs

To disable faults, set *disableInjections* to `True`. To play with other types of FI, make changes to the corresponging YAML file.

The description of each model follows:

**nn-mnist.py**		- A basic neural network architecture with 4 hidder layers to classify MNIST digits.
**cnn-mnist.py**	- A basic convolutional neural network architecture to classify MNIST digits.

Update README here as more models are benchmarked.
