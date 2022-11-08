import pickle
from getDictionary import get_dictionary
import pickle


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']
# note: due to implementation issue, my x and y cooridnate in dictionary might be flipped.
# -----fill in your implementation here --------

output = open('random_dict.pkl', 'wb')
random_dict = get_dictionary(train_imagenames, 50, 100, 'Random')
pickle.dump(random_dict, output)
output.close()

output = open('harris_dict.pkl', 'wb')
harris_dict = get_dictionary(train_imagenames, 50, 100, 'Harris')
pickle.dump(harris_dict, output)
output.close()


# ----------------------------------------------



