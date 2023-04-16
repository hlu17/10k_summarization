"""
This code performs the following tasks:

1. Imports the required libraries and sets the current working directory.
2. Defines command-line arguments for the script.
3. Defines two functions: load_data() to load train or test data from pickle files, and transform_format() to transform the data format and save it as text files.
4. Loads train and test data, transforms their formats, and saves them as text files.
5. Renames label files according to a new naming convention.

"""


# Import required libraries
import os
import json
import torch
import logging
import warnings
import argparse
import transformers
import pandas as pd
from tqdm import trange
from transformers import BartTokenizer, BartForConditionalGeneration

# Ignore warnings and set verbosity level for Transformers library
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# Set the current working directory
os.chdir("")

# Import custom modules
from test_model import *
from test_utils import *

# Import additional libraries
from sklearn.model_selection import train_test_split
import pickle

# Define command-line arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data", type=str)
args = parser.parse_args()

# Function to load train or test data from pickle files
def load_data(dataset="train"):
    """
    Load data from train or test pickle files.

    Params:
    dataset (str): Either "train" or "test" (default: "train")

    Returns:
    pandas.DataFrame: Loaded data as a DataFrame
    """
    # Load train data
    if dataset == "train":
        with open(f"../{args.dataset}/train.pkl", "rb") as f:
            data = pickle.load(f, encoding="latin1")
        return data
    
    # Load test data
    if dataset == "test":
        with open(f"../{args.dataset}/test.pkl", "rb") as f:
            data = pickle.load(f, encoding="latin1")
        return data

# Function to transform data format and save as text files
def transform_format(data, flag):
    """
    Transform data format and save as text files in the corresponding train or test directory.

    Params:
    data (pandas.DataFrame): Data to be transformed and saved
    flag (str): Either "train" or "test"

    Returns:
    None
    """
    for index, row in data.iterrows():
        # Create file name based on the data ID
        file_name = str(row["id"]) + ".txt"
        text = row["text"]
        
        # Save train data as text files
        if flag == "train":
            with open(f"../{args.dataset}/Training_set_new/{file_name}", "w", encoding="utf-8") as f:
                f.write(text)
        # Save test data as text files
        else:
            with open(f"../{args.dataset}/Test_set_new/{file_name}", "w", encoding="utf-8") as f:
                f.write(text)

# Rename label files
import os

os.chdir(os.path.join(f"../{args.dataset}", "Labels_new"))
files = os.listdir()
for file in files:
    filename, ext = os.path.splitext(file)
    if "_" not in filename or not filename.split("_")[0].isnumeric():
        continue
    filename = filename.split("_")[0]
    new_filename = filename + "_1" + ext
    os.rename(file, new_filename)
