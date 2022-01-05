import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DeepPurpose.utils import *

'''
Author: Frankie Fan

Acknowledgement:
We filter the dataset based on column:
    Number of Protein Chains in Target (>1 implies a multichain complex)
    Ligand SMILES
    PubChem CID
    UniProt (SwissProt) Primary ID of Target Chain
    Ligand InChI
    Ki (nM)
    IC50 (nM)
    Kd (nM)
    EC50 (nM)
'''


def preprocessing_BindingDB(path = None, df = None, y = 'Kd', binary = False, convert_to_log = True, threshold = 30, filter_rough_value = True, save_filtered_dataset_to_file = None, max_smiles_length = None, max_protein_length = None):
    '''
    The y is one of these types: Kd, IC50, Ki, EC50
    '''
    if df is not None:
        print('Loading Dataset from the pandas input...')
    else:
        print('Loading Dataset from path...')
        if not os.path.isfile(path):
            print('File not exist: ' + path)
            return
        df = pd.read_csv(path, sep = '\t', error_bad_lines=False)
    print('Beginning Processing...')
    df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    df = df[df['Ligand SMILES'].notnull()]

    if y == 'Kd':
        idx_str = 'Kd (nM)'
    elif y == 'IC50':
        idx_str = 'IC50 (nM)'
    elif y == 'Ki':
        idx_str = 'Ki (nM)'
    elif y == 'EC50':
        idx_str = 'EC50 (nM)'
    else:
        print('select Kd, Ki, IC50 or EC50')
        return

    df_want = df[df[idx_str].notnull()]
    df_want = df_want[['BindingDB Reactant_set_id', 'Ligand InChI', 'Ligand SMILES', \
                       'PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', \
                       'BindingDB Target Chain  Sequence', idx_str]]
    df_want.rename(columns={'BindingDB Reactant_set_id':'ID',
                            'Ligand SMILES':'SMILES',
                            'Ligand InChI':'InChI',
                            'PubChem CID':'PubChem_ID',
                            'UniProt (SwissProt) Primary ID of Target Chain':'UniProt_ID',
                            'BindingDB Target Chain  Sequence': 'Target Sequence',
                            idx_str: 'Label'},
                   inplace=True)

    if filter_rough_value:
        #         df_want = df_want[df_want['Label'].astype(str).str.contains('>|<') == False]
        df_want = df_want[~df_want['Label'].str.contains('>|<', na=False)]
    else:
        df_want['Label'] = df_want['Label'].str.replace('>', '')
        df_want['Label'] = df_want['Label'].str.replace('<', '')
    df_want['Label'] = df_want['Label'].astype(float)

    # have at least uniprot or pubchem ID
    df_want = df_want[df_want.PubChem_ID.notnull() | df_want.UniProt_ID.notnull()]
    df_want = df_want[df_want.InChI.notnull()]

    df_want = df_want[(df_want.Label > 0) & (df_want.Label <= 10000000.0)]

    # filter the records with too long SMILES or protein sequences
    if max_smiles_length is not None:
        df_want = df_want[df_want.SMILES.apply(lambda x: len(str(x))<max_smiles_length)]
    if max_protein_length is not None:
        df_want = df_want[df_want['Target Sequence'].apply(lambda x: len(str(x))<max_protein_length)]

    print('There are ' + str(len(df_want)) + ' drug target pairs.')

    if binary:
        print('Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter')
        y = [1 if i else 0 for i in df_want.Label.values < threshold]
    else:
        if convert_to_log:
            print('Default set to logspace (nM -> p) for easier regression')
            y = convert_y_unit(df_want.Label.values, 'nM', 'p')
        else:
            y = df_want.Label.values

    if save_filtered_dataset_to_file is not None:
        print('Saving filtered dataset to path...')
        df_want['LABEL_CONVERTED'] = y
        df_want.to_csv(save_filtered_dataset_to_file, sep = '\t', index=False)

    return df_want.SMILES.values, df_want['Target Sequence'].values, np.array(y)


def filter_data(path = None, y = 'Kd'):
    '''
    The y is one of these types: Kd, IC50, Ki, EC50
    '''
    print('Loading Dataset from path...')
    if not os.path.isfile(path):
        print('File not exist: ' + path)
        return
    df = pd.read_csv(path, sep = '\t', error_bad_lines=False)

    print('Beginning Processing...')
    df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    df = df[df['Ligand SMILES'].notnull()]

    if y == 'Kd':
        idx_str = 'Kd (nM)'
    elif y == 'IC50':
        idx_str = 'IC50 (nM)'
    elif y == 'Ki':
        idx_str = 'Ki (nM)'
    elif y == 'EC50':
        idx_str = 'EC50 (nM)'
    else:
        print('select Kd, Ki, IC50 or EC50')
        return

    df_want = df[df[idx_str].notnull()]
    df_want = df_want[['BindingDB Reactant_set_id', 'Ligand InChI', 'Ligand SMILES', \
                       'PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', \
                       'BindingDB Target Chain  Sequence', idx_str]]
    df_want.rename(columns={'BindingDB Reactant_set_id':'ID',
                            'Ligand SMILES':'SMILES',
                            'Ligand InChI':'InChI',
                            'PubChem CID':'PubChem_ID',
                            'UniProt (SwissProt) Primary ID of Target Chain':'UniProt_ID',
                            'BindingDB Target Chain  Sequence': 'Target Sequence',
                            idx_str: 'Label'},
                   inplace=True)

    df_want = df_want[~df_want['Label'].str.contains('>|<', na=False)]
    df_want['Label'] = df_want['Label'].astype(float)
    df_want = df_want[(df_want.Label > 0.01) & (df_want.Label < 10000000.0)]

    df_want = df_want[~df_want['Target Sequence'].str.contains('[X0-9a-z]', regex=True)]

    print('There are ' + str(len(df_want)) + ' drug target pairs.')
    return df_want


def process_duplicates(drugs, targets, labels):
    '''
        Return
            1. basic dataset: Drug, Target, Label, Length of Drug, Length of Target
            2. distinct Drug&Target&Label pairs together with duplicate times: Drug, Target, Label, Length of Drug, Length of Target
            3. distinct Drug&Target pairs together with duplicate times: Drug, Target, (size)
            4. distinct Drug&Target pairs which have duplicate Label values: Drug, Target
            5. records relating duplicate Drug&Target pairs: Drug, Target, Label, Length of Drug, Length of Target
    '''
    df_compact = pd.DataFrame(zip(drugs, targets, labels), columns=['Drug','Target','Label'])
    df_compact['Length of Drug'] = df_compact['Drug'].apply(lambda x: len(str(x)))
    df_compact['Length of Target'] = df_compact['Target'].apply(lambda x: len(str(x)))
    df_compact_distinct = df_compact.groupby(['Drug','Target','Label']).size()
    print('There are ' + str(len(df_compact_distinct)) + ' distinct Drug&Target&Label pairs out of ' + str(len(df_compact)) + ' total records.')

    df_group = df_compact.groupby(['Drug','Target']).size()
    df_dup_ids = df_group[df_group>1].reset_index()[['Drug','Target']]
    print('There are ' + str(len(df_dup_ids)) + ' distinct Drug&Target pairs out of ' + str(len(df_group)) + ' distinct Drug&Target pairs have duplicate records.')

    df_compact_dup = pd.merge(df_dup_ids,df_compact,on=['Drug','Target'])
    print('There are ' + str(len(df_compact_dup)) + ' records relating duplicate Drug&Target pairs.')

    return df_compact, df_compact_distinct, df_group, df_dup_ids, df_compact_dup

def process_label_duplicates(drugs, targets, labels):
    print('Set label to logspace (nM -> p) for easier regression')
    y = convert_y_unit(labels.values, 'nM', 'p')
    df_p = pd.DataFrame(zip(drugs, targets, y), columns=['Drug','Target','Label'])
    df_p = df_p.groupby(['Drug','Target'])['Label'].mean().reset_index()
    print('There are ' + str(len(df_p)) + ' drug target pairs after (nM -> p) and group&mean.')
    return df_p

def cutoff_by_length(drugs, targets, labels):
    df_p = pd.DataFrame(zip(drugs, targets, labels), columns=['Drug','Target','Label'])
    df_p['Length of Drug'] = drugs.apply(lambda x: len(str(x)))
    df_p['Length of Target'] = targets.apply(lambda x: len(str(x)))
    df_p = df_p[(df_p['Length of Drug'] >= 5) & (df_p['Length of Drug'] <= 200) & (df_p['Length of Target'] >= 50) & (df_p['Length of Target'] <= 2000)]
    print('There are ' + str(len(df_p)) + ' drug target pairs after cutoff.')
    return df_p

def preprocessing_BindingDB_kdki(path = None):

    print('Loading Kd Dataset from path...')
    df_kd = filter_data(path, y='Kd')
    df_kd_compact, _, _, _, _ = process_duplicates(df_kd['SMILES'], df_kd['Target Sequence'], df_kd['Label'])

    print('Loading Ki Dataset from path...')
    df_ki = filter_data(path, y='Ki')
    df_ki_compact, _, _, _, _ = process_duplicates(df_ki['SMILES'], df_ki['Target Sequence'], df_ki['Label'])

    print('Combine Kd and Ki Dataset and preprocess them...')
    df_kdki = pd.concat([df_kd_compact, df_ki_compact])
    df_kdki_compact, _, _, _, _ = process_duplicates(df_kdki['Drug'], df_kdki['Target'], df_kdki['Label'])

    df_p = process_label_duplicates(df_kdki_compact['Drug'], df_kdki_compact['Target'], df_kdki_compact['Label'])
    df_p = cutoff_by_length(df_p['Drug'], df_p['Target'], df_p['Label'])

    return df_p
