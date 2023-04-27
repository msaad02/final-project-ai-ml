import keras
import numpy as np
import pandas as pd
import csv
import tensorflow as tf

import utils

class DataSequence(keras.utils.Sequence):
  
    def __init__(self, file, batch_size):
        self.file = file
        self.batch_size = batch_size
      
    def __len__(self):
        #total_length = sum(1 for row in open(self.file))
        total_length = 4000000
        return int(np.ceil(total_length / self.batch_size))
    
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.file, skiprows=idx*self.batch_size, nrows=self.batch_size)
      
        x = np.ndarray(shape=(self.batch_size,832))
        y = np.array(df.iloc[:,1])
      
        #have preprocessing here right now for testing, should save processed scores in file
        # once we decide how to do this
        for i, f in enumerate(df.iloc[:,0]):
            x[i] = utils.vectorize(f)
        
        y = utils.preprocess_scores(y)
        #print(x.shape, y.shape)
        #print(idx)
        
        return (x, y)