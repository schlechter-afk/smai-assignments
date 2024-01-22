import numpy as np
import random

# Load the entire dataset
data = np.load('data_train.npy', allow_pickle=True)

test_ratio = 0.2

# Calculate the number of examples for test data
num_test_examples = int(len(data) * test_ratio)

test_indices = np.random.choice(len(data), size=num_test_examples, replace=False)
test_data = data[test_indices]

train_data = np.delete(data, test_indices, axis=0)

# Save the test data to a new file
np.save('test_data.npy', test_data)

# Save the remaining data as the new train data
np.save('train_data.npy', train_data)

# print(f"Generated test data with {num_test_examples} examples and saved as test_data.npy.")
# print(f"Updated train data with {len(train_data)} examples and saved as train_data.npy.")