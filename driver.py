import matplotlib.pyplot as plt
import numpy as np

import ioHandler
import neuralNet

def main():
	print("program start")
	print("load data")
	# load data from json files, convert 2 character letter matrix to 0/1, validate data
	training_data, testing_data = ioHandler.load_data()

	print(training_data)

	# preprocess training and testing set
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	for i in training_data['data']:
		train_x.append(i['matrix'])
		train_y.append(i['label'])

	for i in testing_data['data']:
		test_x.append(i['matrix'])
		test_y.append(i['label'])

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	train_x = train_x
	test_x = test_x
	train_y = train_y.reshape((1, train_y.shape[0]))
	test_y = test_y.reshape((1, test_y.shape[0]))

	print(train_x)
	print("")
	print(train_y)
	print("")
	print(test_x)
	print("")
	print(test_y)

	print ("Train X shape: " + str(train_x.shape))
	print ("Train Y shape: " + str(train_y.shape))
	print ("Test X shape: " + str(test_x.shape))
	print ("Test Y shape: " + str(test_y.shape))


	iterations = 200
	learning_rate = 0.005
	# call neural network model
	train_pred_y, test_pred_y, costs = neuralNet.model(train_x, train_y, test_x, test_y, iterations, learning_rate)

	print(train_pred_y)
	print(test_pred_y)

	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

if __name__ == "__main__":
	main()