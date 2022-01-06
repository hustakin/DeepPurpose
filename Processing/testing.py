import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
import Processing.dataset_filter as processors

'''
Author: Frankie Fan

Testing a given dataset based on a pre-trained model
'''


def batch_predict(path, name, X_drug, X_target, y):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'CNN'
    target_encoding = 'CNN'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
    print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
              + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
    print('--- Finish Testing ' + name + ' ---')
