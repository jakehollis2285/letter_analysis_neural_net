The important logical files in this project are:

	driver.py
		- main application entrypoint
		- run the application using:
			python3 driver.py
		- edit this file to change number of iterations and learning rate for each neural network runthrough

	ioHandler.py
		- implements the data loading and preprocessing functions
		- SHOULD NOT be edited

	neuralNet.py
		- implements neural network with no hidden layers
		- initialize weight matrix
		- vectorize input / output
		- run gradient descent to update weight matrix
		- create one hot vector representation of A and Y to compare
		- get training set and testing set results
		- SHOULD NOT be edited

This application accepts input in the following form:

	- a 5x5 flattened to a 1x25 array of character symbols,
	- only 2 symbols can be used in any given array,
	- example input is given below:

	This is an "A"
		..#..		#####
		.#.#.		#...#
		.###.		#####
		#...#		#...#
		#...#		#...#

	This is an "X"
		#...#		#...#
		.#.#.		#...#
		..#..		.###.
		.#.#.		#...#
		#...#		#...#

	This is an "O"
		..#..		.###.
		.#.#.		#...#
		#...#		#...#
		.#.#.		#...#
		..#..		.###.

For each of the above examples, the input will be flattened as a development step. To see input examples, view ./data/training_set.json or ./data/testing_set.json

There are more examples of the character set I have used in ./data/char_set_examples.png; the character set is a 5x5 pixel characterset with all english letters, you can see flattened examples in ./data/alphabet.json. The file alphabet.json is incomplete since the task in this project was to be able to differentiate between the letters "X", "O", and an arbitrary third letter (in my case "A").

The files ./data/training_set.json and ./data/testing_set.json are the INPUT files for the neural network application. Replace these files to define new training and testing sets. These files are JSON spec documents and the data layout MUST be followed.

The json documents are in the following form:

	{
		"data": [ // list of examples
			{
				"matrix": "######...#######...##...#", 	// flat text input remove whitespace and newline characters from text representations
				"label": "A"							// label of the example cooresponing to label in "labels"
			},
			{
				"matrix": "#...##...#.###.#...##...#",
				"label": "X"
			},
			{
				"matrix": "######...##...##...######",
				"label": "O"
			}
		],
		"labels": ["A", "X", "O"] // set of characters present in the data
	}

The data format above is generic, "label" can be any single character, "matrix" must be 5x5 flattened to 1x25
The data is preprocessed in ioHandler.py and converted into a 0 / 1 representation (i.e. the first character (and all characters identical to the first) in the array are replaced with 0, all other characters in the array are replaced by 1). Likewise, the "labels" are converted into a representation of 0 - 25 where 0 is the first letter in the training set and 25 is the last letter. This is assuming the english alphabet if there are more unique labels that 26, the neural network should still function, but this behavior is undefined.

