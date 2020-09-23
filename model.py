# Import libraries
import numpy as np
import pandas as pd

# Load dataset
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# The preprocessor function will be implement here...
from preprocessing_data import processor_data

clean_data = processor_data(train_data, test_data)
