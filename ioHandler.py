import json

def load_data():
	tr = open("./data/training_set.json")
	te = open("./data/testing_set.json")
	training_data = json.load(tr)
	testing_data = json.load(te)

	training_data = validate_data("training_data", training_data, 0)	
	testing_data = validate_data("testing_data", testing_data, 0)

	return training_data, testing_data

def validate_data(name, data, print_data):
	new_labels = range(len(data['labels']))

	for i in data["data"]:
		if(len(i["matrix"]) != 25):
			print(name, " malformed; check index ", i)
			quit()
		i['matrix'] = parse_matrix(name, i, i['matrix'], print_data, i['label'])
		if(i['label'] not in data['labels']):
			print(name, " malformed; label not in label set, check index ", i)
			quit()
		i['label'] = new_labels[data['labels'].index(i['label'])]

	return data

def parse_matrix(name, index, li, print_data, label):
	li = list(li)
	tmp_a = None
	tmp_b = None
	for i in li:
		if(tmp_a == None and tmp_b == None):
			tmp_a = i

		elif(tmp_b == None and i != tmp_a):
			tmp_b = i

		elif(i == tmp_a or i == tmp_b or tmp_a == None or tmp_b == None):
				continue
		else:
			print("matrix in ", name, " contains more than 2 characters; check index ", i)
			print(tmp_a)
			print(tmp_b)
			print(li)
			quit()

	li = [0 if x == tmp_a else x for x in li]
	li = [1 if x == tmp_b else x for x in li]
	if(print_data):
		print(li)
		print(label)

	return li
