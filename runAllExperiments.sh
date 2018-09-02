#### run All the experimental tests in the paper multiple times


### Specify how much times you want to run these tests, 
### e.g., 10 times ==> for i in {1..10}

for i in {1..100}
do
	# run logistic regression model for 6 dataset, results for each test will be stored separately.
	python ./experimentalTest/adult_logistic_regression.py ./experimentalTest/accuracyResults/adult_LR.csv
	python ./experimentalTest/credit_logistic_regression.py ./experimentalTest/accuracyResults/credit_LR.csv
	python ./experimentalTest/marketing_logistic_regression.py ./experimentalTest/accuracyResults/marketing_LR.csv
	python ./experimentalTest/mnist_logistic_regression.py ./experimentalTest/accuracyResults/mnist_LR.csv
	python ./experimentalTest/survive_logistic_regression.py ./experimentalTest/accuracyResults/survive_LR.csv
	python ./experimentalTest/zoo_logistic_regression.py ./experimentalTest/accuracyResults/zoo_LR.csv


	# run nearest neighbour model for 6 dataset, results for each test will be stored separately.
	python ./experimentalTest/adult_nearest_neighbour.py ./experimentalTest/accuracyResults/adult_kNN.csv
	python ./experimentalTest/credit_nearest_neighbour.py ./experimentalTest/accuracyResults/credit_kNN.csv
	python ./experimentalTest/marketing_nearest_neighbour.py ./experimentalTest/accuracyResults/marketing_kNN.csv
	python ./experimentalTest/mnist_nearest_neighbour.py ./experimentalTest/accuracyResults/mnist_kNN.csv
	python ./experimentalTest/survive_nearest_neighbour.py ./experimentalTest/accuracyResults/survive_kNN.csv
	python ./experimentalTest/zoo_nearest_neighbour.py ./experimentalTest/accuracyResults/zoo_kNN.csv
	

	# run neural network model for 6 dataset, results for each test will be stored separately.
	python ./experimentalTest/adult_nn.py ./experimentalTest/accuracyResults/adult_nn.csv
	python ./experimentalTest/credit_nn.py ./experimentalTest/accuracyResults/credit_nn.csv
	python ./experimentalTest/marketing_nn.py ./experimentalTest/accuracyResults/marketing_nn.csv
	python ./experimentalTest/mnist_nn.py ./experimentalTest/accuracyResults/mnist_nn.csv
	python ./experimentalTest/survive_nn.py ./experimentalTest/accuracyResults/survive_nn.csv
	python ./experimentalTest/zoo_nn.py ./experimentalTest/accuracyResults/zoo_nn.csv
	
done
