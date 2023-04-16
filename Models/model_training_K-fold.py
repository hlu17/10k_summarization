"""
This code performs the following tasks to train a BART model for text summarization:

1.Set up the device for training (GPU or CPU).
2.Load the training data using the load_data function.
3.Define the hyperparameters for source and target sequence lengths.
4.Initialize the BART model, optionally with custom attention, and move it to the device.
5.Log the model details.
6.Define the optimizer for the model (AdamW).
7.Define the tokenizer for the model.
8.Define the task-specific prefix for the tokenizer.
9.Set up variables to keep track of loss values during training and validation.
10.Set up cross-validation using K-Fold.
    For each fold in the cross-validation:
        a. Define train and validation sets for the fold.
        b. Loop over the epochs.
            i. Set the model to training mode.
            ii. Loop over mini-batches of the training dataset.
                1. Encode the inputs and targets.
                2. Perform a forward pass through the model.
                3. Perform a backward pass to calculate gradients.
                4. Update the model parameters using the optimizer.
                5. Accumulate the loss values and log them periodically.
            iii. Set the model to evaluation mode.
            iv. Loop over mini-batches of the validation dataset.
                1. Encode the inputs and targets.
                2. Perform a forward pass through the model.
                3. Accumulate the loss values and log them periodically.
            v. Save the model if the validation loss is the best encountered so far.
            vi. Generate summaries for the validation set and calculate evaluation metrics like ROUGE and BERTScore.
12. Log that the training is complete.
"""

import os
import json
import torch
import logging
import warnings
import argparse
import transformers
import pandas as pd
from tqdm import trange
from transformers import BartTokenizer,BartForConditionalGeneration
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.chdir("")
from test_model import *
from test_utils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-t","--do_train",default=True,action="store_true")
parser.add_argument("-p","--do_predict",default=True,action="store_true")
parser.add_argument("-a","--add_attention",default=False,action="store_true")
parser.add_argument("-d","--dataset",default="data",type=str)
parser.add_argument("-m","--model_name",default="facebook/bart-large-cnn",type=str)
parser.add_argument("-u","--using_model",default="../models/facebook/bart-large-cnn/",type=str)
parser.add_argument("-e","--epoch", default=10, type=int)
parser.add_argument("-bs","--batch_size", default=2, type=int)
parser.add_argument("-bp","--batch_print", default=1, type=int)
parser.add_argument("-cn","--cuda_num",default=0,type=int)
parser.add_argument("-o","--out_file_name", default="bart_sum_new_att",type=str)
parser.add_argument("-l","--log_name", default="bart_sum_new_att",type=str)
parser.add_argument("-ma","--model_att", default="model_att",type=str)
args = parser.parse_args()


# Function to load train or test data from CSV files
def load_data(dataset = "train"):
    if dataset == "train":
        return pd.read_csv(f"../{args.dataset}/train_new.csv")
    if dataset == "test":
        return pd.read_csv(f"../{args.dataset}/test_new.csv")

# Function to set up logging for the script
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

logger = get_logger(filename=f"../logs/{args.log_name}.log")

# Log start of the new session and input arguments
logger.info(f"-------------------------------------------new_log------------------------------------------")
logger.info(f"input args:{args}")
# List of keywords for the custom attention mechanism
key_words=["sales", "revenues", "profit", "income", "operation",
                                                 "ebitda", "cash", "flow", "capex", "leverage",
                                                 "equity", "debt",  "borrowings", "indebtedness",
                                                 "performance", "profitability", "capital",
                                                 "expenditures", "liquidity", "ebit"]

'''
create a function to train bart model
'''
if args.do_train:
    device = torch.device(f"cuda:{args.cuda_num}")
    # load data
    data = load_data("train")
    # #random select train set
    # data=data.sample(n=50, random_state=1).reset_index()
    # define hyperparameters
    max_source_length = 512
    max_target_length = 512
    # define model
    if args.add_attention:
        model = MyBartForConditionalGeneration.from_pretrained(args.model_name, key_words).to(device)
    else:
        model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)
    logger.info(model)
    
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # define tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    # define task-specific prefix
    task_prefix = "summarize: "

    # define loss function
    running_loss = 0.0
    # define loss stack
    loss_stack = []
    best_loss = 100

    # define loss function
    valid_running_loss = 0.0
    # define loss stack
    valid_loss_stack = []
    valid_best_loss = 100
    

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # total_loss = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
        # define train and validation sets for this fold
        train = data.loc[train_idx].reset_index()
        valid = data.loc[valid_idx].reset_index()

        # define training loop
        for epoch in range(args.epoch): # loop over the dataset multiple times
            model.train()
            for i in range(0,len(train["text"]),args.batch_size): # loop over mini-batches of the dataset           
                # encode the inputs
                input_sequences = [train["text"][i] for i in range(i,min(i+args.batch_size,len(train["text"])))]
                output_sequence = [train["sum"][i] for i in range(i,min(i+args.batch_size,len(train["sum"])))]
                encoding = tokenizer([task_prefix + sequence for sequence in input_sequences],
                                    padding='longest',
                                    max_length=max_source_length,
                                    truncation=True)
                input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
                attention_mask = torch.tensor(attention_mask).to(device)
                input_ids = torch.tensor(input_ids).to(device)
                # encode the targets
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(output_sequence, max_length=max_target_length, padding="longest", truncation=True, return_tensors="pt")
                labels.to(device)
                optimizer.zero_grad()
                # forward pass
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels["input_ids"]).loss
                # backward pass
                loss.backward()
                # update parameters
                optimizer.step()
                # accumulate loss
                running_loss += loss.item()
                # print(running_loss)
                if i % args.batch_print == 0 and i != 0:
                    logger.info(f"fold:{fold},epoch:{epoch}/{args.epoch}, batch:{i//args.batch_size}/{len(train)//args.batch_size}, loss:{running_loss/args.batch_print}")
                    loss_stack.append(running_loss/args.batch_print)
                    running_loss = 0.0
            
            model.eval()
            # valid_loss = 0.0
            with torch.no_grad():
                for i in range(0,len(valid["text"]),args.batch_size): # loop over mini-batches of the dataset           
                    # encode the inputs
                    input_sequences = [valid["text"][i] for i in range(i,min(i+args.batch_size,len(valid["text"])))]
                    output_sequence = [valid["sum"][i] for i in range(i,min(i+args.batch_size,len(valid["sum"])))]
                    encoding = tokenizer([task_prefix + sequence for sequence in input_sequences],
                                        padding='longest',
                                        max_length=max_source_length,
                                        truncation=True)
                    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
                    attention_mask = torch.tensor(attention_mask).to(device)
                    input_ids = torch.tensor(input_ids).to(device)
                    # encode the targets
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(output_sequence, max_length=max_target_length, padding="longest", truncation=True, return_tensors="pt")
                    labels.to(device)
                    # forward pass
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels["input_ids"]).loss
                    # accumulate loss
                    valid_running_loss += loss.item()
                    # print(running_loss)
                    if i % args.batch_print == 0 and i != 0:
                        logger.info(f"fold:{fold},epoch:{epoch}/{args.epoch}, batch:{i//args.batch_size}/{len(valid)//args.batch_size}, loss:{valid_running_loss/args.batch_print}")
                        valid_loss_stack.append(valid_running_loss/args.batch_print)
                        valid_running_loss = 0.0

                # when best loss save the model
                if min(valid_loss_stack) < valid_best_loss:
                    valid_best_loss = min(valid_loss_stack)
                    model.save_pretrained(f"../models/{args.model_att}/{args.using_model}")
                    logger.info(f"******{args.using_model} model saved******")
                    loss_stack.clear()
                    valid_loss_stack.clear()

                    predict_list=[]
                    true_list=[]
                    for i in trange(len(valid["text"])):
                        dic = {"text":valid["text"][i],"sum":valid["sum"][i]}
                        params = {"decoder_start_token_id":0,"early_stopping":False,"no_repeat_ngram_size":0,"length_penalty": 0,"num_beams":4,"use_cache":True}
                        summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=max_target_length,**params)
                        predict_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        if predict_text.startswith("new : "):
                            predict_text = predict_text[6:]
                        if not predict_text=='':
                            dic["predict"] = predict_text
                            predict_list.append(dic["predict"])
                            true_list.append(dic["sum"])
                    try:
                        rouge1,rouge2,rougel,bertscore=calculate_metrics(true_list,predict_list)
                        logger.info(f"ROUGE-1: {rouge1}")
                        logger.info(f"ROUGE-2: {rouge2}")
                        logger.info(f"ROUGE-L: {rougel}")
                        logger.info(f"BERTScore: {bertscore}")
                    except:
                        a=1

    logger.info("train done")





'''a predict function for bart_summary'''
if args.do_predict:
    device = torch.device(f"cuda:{args.cuda_num}")
    test = load_data("test")
    # #random select test set
    # test=test.sample(n=10, random_state=1).reset_index()
    # define hyperparameters
    max_source_length = 512
    max_target_length = 512
    # load model 
    if args.add_attention:
        model = MyBartForConditionalGeneration.from_pretrained(f"../models/{args.using_model}", key_words).to(device)
    else:
        model = BartForConditionalGeneration.from_pretrained(f"../models/{args.model_name}").to(device)
    # define tokenizer
    # define task-specific prefix
    task_prefix = "summarize: "
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model.eval()
    predict_list=[]
    true_list=[]
    for i in trange(len(test["text"])):
        dic = {"id":test["id"][i],"text":test["text"][i]}
        encoding = tokenizer(task_prefix + test["text"][i],
                            padding='longest',
                            max_length=max_source_length,
                            truncation=True,
                            return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        params = {"decoder_start_token_id":0,"early_stopping":False,"no_repeat_ngram_size":0,"length_penalty": 0,"num_beams":4,"use_cache":True}
        summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=max_target_length,**params)
        predict_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if predict_text.startswith("new : "):
            predict_text = predict_text[6:]
        dic["predict"] = predict_text
        with open(f"../results/{args.out_file_name}/{dic['id']}.txt","w") as f:
            f.write(json.dumps(predict_text)+"\n")
        predict_list.append(dic["predict"])

    d={}
    for output in predict_list:
        keyword_counts = {keyword: output.count(keyword) for keyword in key_words}
        for keyword, count in keyword_counts.items():
            if not keyword in d:
                d[keyword]=count
            else:
                d[keyword]+=count
    logger.info(f"{d}")
    logger.info("predict done")
