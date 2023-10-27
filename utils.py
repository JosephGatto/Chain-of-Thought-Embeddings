import random 
import numpy as np 
import torch 
import tensorflow as tf
from transformers import set_seed 
import os 
from sklearn.metrics import classification_report
def get_f1(labels, predictions):
  '''
  This is how F1 is defined for Tweet-Stance in original work
  '''
  c_report = classification_report(labels, predictions, output_dict=True)
  f1_against = c_report['1']['f1-score']
  f1_favor = c_report['2']['f1-score']
  tweeteval_result = (f1_against+f1_favor) / 2
  return tweeteval_result

def parse_cot(x, debug = False):

  '''
  ChatGPT String Output Parser 
  '''
  x = x.lower()
  if '[yes]' in x:
    if debug:
      print(x, '-> favor')
    return 2
  elif '[no]' in x or '[against]' in x:
    if debug:
      print(x, '-> against')
    return 1
  elif '[none]' in x or '[neutral]' in x:
    if debug:
      print(x, '-> neutral')
    return 0
  else:
    raise ValueError(f'Could not parse {x}')


def set_all_seeds(seed):
  
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


  np.random.seed(seed)
  tf.random.set_seed(seed)
  random.seed(seed)
  set_seed(seed)

  os.environ['PYTHONHASHSEED'] = str(seed)




