#!/bin/bash

# python generate_random_test_data.py # remove this when submitting

# Check if the required arguments are provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <test_data_file.npy>"
    exit 1
fi

# Get the path to the test data file
test_data_file="$1"

# Check if the test data file exists
if [ ! -f "$test_data_file" ]; then
    echo "Error: Test data file not found."
    exit 1
fi

# Call the Python script to evaluate KNN and get results
python testscript.py "$test_data_file"
