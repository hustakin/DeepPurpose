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


def batch_predict(path, name, X_drug, X_target, y, repurposing_mode = False):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'CNN'
    target_encoding = 'CNN'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    if repurposing_mode:
        pred = model.predict(test, repurposing_mode = True, test = True)
        print('--- Finish Predicting ' + str(len(pred)) + ' records ---')
        return pred
    else:
        mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
        print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
        print('--- Finish Testing ' + name + ' ---')

# below added by YS 4Feb2022        
def batch_predict_morgan_cnn(path, name, X_drug, X_target, y, repurposing_mode = False):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'Morgan'
    target_encoding = 'CNN'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    if repurposing_mode:
        pred = model.predict(test, repurposing_mode = True, test = True)
        print('--- Finish Predicting ' + str(len(pred)) + ' records ---')
        return pred
    else:
        mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
        print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
        print('--- Finish Testing ' + name + ' ---')

def batch_predict_daylight_aac(path, name, X_drug, X_target, y, repurposing_mode = False):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'Daylight'
    target_encoding = 'AAC'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    if repurposing_mode:
        pred = model.predict(test, repurposing_mode = True, test = True)
        print('--- Finish Predicting ' + str(len(pred)) + ' records ---')
        return pred
    else:
        mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
        print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
        print('--- Finish Testing ' + name + ' ---')
        
def batch_predict_espf_espf(path, name, X_drug, X_target, y, repurposing_mode = False):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'ESPF'
    target_encoding = 'ESPF'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    if repurposing_mode:
        pred = model.predict(test, repurposing_mode = True, test = True)
        print('--- Finish Predicting ' + str(len(pred)) + ' records ---')
        return pred
    else:
        mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
        print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
        print('--- Finish Testing ' + name + ' ---')
        
def batch_predict_morgan_aac(path, name, X_drug, X_target, y, repurposing_mode = False):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'Morgan'
    target_encoding = 'AAC'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    if repurposing_mode:
        pred = model.predict(test, repurposing_mode = True, test = True)
        print('--- Finish Predicting ' + str(len(pred)) + ' records ---')
        return pred
    else:
        mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
        print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
        print('--- Finish Testing ' + name + ' ---')
        
def batch_predict_trans_cnn(path, name, X_drug, X_target, y, repurposing_mode = False):
    print('Loading pre-trained model from path: ' + path)
    model = models.model_pretrained(path)
    drug_encoding = 'Transformer'
    target_encoding = 'CNN'
    test = data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
    if repurposing_mode:
        pred = model.predict(test, repurposing_mode = True, test = True)
        print('--- Finish Predicting ' + str(len(pred)) + ' records ---')
        return pred
    else:
        mse, r2, p_val, CI, logits, loss_val = model.predict(test, repurposing_mode = False, test = True)
        print('[' + name + '] Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
        print('--- Finish Testing ' + name + ' ---')
# above added by YS 4Feb2022  