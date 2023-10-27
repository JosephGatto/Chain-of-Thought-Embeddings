

import argparse

from torch import nn
import torch
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback

import datasets
from datasets import Dataset, DatasetDict

import pandas as pd
import numpy as np
import os
from pprint import pprint 

import evaluate
from sklearn.metrics import mean_squared_error, classification_report
from utils import parse_cot, set_all_seeds, get_f1 


def tokenize_function(examples, modality, tokenizer, max_length = 512):

    assert modality in ['text-only', 'cot-only', 'text+cot']
    if modality == 'text-only':
      encoded = tokenizer(examples['topic'], examples['text'], padding = 'max_length', truncation = True, max_length = max_length)
    elif modality == 'cot-only':
      encoded = tokenizer(examples[f'COT'], padding = 'max_length', truncation = True, max_length = max_length)
    elif modality == 'text+cot':
      encoded = tokenizer(examples['topic'] + ': ' + examples['text'], examples[f'COT'], padding = 'max_length', truncation = True, max_length = max_length)
    else:
      raise NotImplementedError()
    encoded['labels'] = examples['label']
    return encoded


def compute_metrics(eval_pred):
      '''
      Stance Detection Eval Metric
      '''
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      c_report = classification_report(labels, predictions, output_dict=True)
      f1_against = c_report['1']['f1-score']
      f1_favor = c_report['2']['f1-score']
      tweeteval_result = (f1_against+f1_favor) / 2
      return {'eval_f1':tweeteval_result}



def run(dataset, modality, seed, num_labels = 3):


  ### Learning rates were chosen after grid search ### 
  learning_rate_dict = {
        'text-only': 2e-5,
        'cot-only': 2e-5,
        'text+cot': 5e-5
    }
  
  MODEL_NAME = "cardiffnlp/twitter-roberta-base-sep2022"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenized_datasets = dataset.map(lambda x: tokenize_function(x, modality = modality, tokenizer=tokenizer), batched=False)
  tokenized_datasets = tokenized_datasets.remove_columns(dataset['test'].to_pandas().columns)
  

  def model_init():

    set_all_seeds(seed)
    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.classifier.apply(model._init_weights)
    return model

  training_args = TrainingArguments(output_dir=f'trained_{modality}',
                                    evaluation_strategy="epoch",
                                    per_device_train_batch_size=16,
                                    num_train_epochs = 10,
                                    load_best_model_at_end=True,
                                    save_strategy='epoch',
                                    learning_rate = learning_rate_dict[modality],
                                    overwrite_output_dir=True,
                                    save_total_limit=1,
                                    fp16=True)


  trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['dev'],
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],

  )

  trainer.train()
  output_f1 = trainer.evaluate(tokenized_datasets['test'])['eval_f1']
  predictions = trainer.predict(tokenized_datasets['test']).predictions.argmax(-1)

  return output_f1, predictions, tokenized_datasets['test']['labels']




def get_llm_scores(dataset):
  test = dataset['test'].to_pandas().copy()
  test['COT-Output-pred'] = test['COT'].apply(parse_cot)
  cot_performance = get_f1(test['label'].tolist(), test['COT-Output-pred'].tolist())
  return cot_performance
    

def main(args):



  root = 'Data/'

  # Build dataset for training
  dataset = DatasetDict({
      'train': Dataset.from_pandas(pd.read_csv(root+'Tweet-Eval-Train-ChatGPT.csv')),
      'dev': Dataset.from_pandas(pd.read_csv(root+'Tweet-Eval-Dev-ChatGPT.csv')),
      'test': Dataset.from_pandas(pd.read_csv(root+'Tweet-Eval-Test-ChatGPT.csv')),
  })


  cot_performance = get_llm_scores(dataset)
  pprint({'COT Performance (ChatGPT)': cot_performance})
  f1_score, predictions, labels = run(dataset, args.modality, seed=args.seed)
  print(f"F1 Score: {f1_score}")


if __name__ == '__main__':

  # Define the ArgumentParser
  parser = argparse.ArgumentParser()

  # Add arguments
  parser.add_argument("modality")
  parser.add_argument("seed", type=int, default=1)

  # Indicate end of argument definitions and parse args
  args = parser.parse_args()
  assert args.modality in ['text-only', 'cot-only', 'text+cot']
  main(args)
  