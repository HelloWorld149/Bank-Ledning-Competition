import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer as MICE
from collections import Counter
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import random
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#read the file name
def load_data(file_name, test_file_name):
    
    dataset = pd.read_csv(file_name)
    dataset_cp = dataset.copy()
    y = dataset['loan_paid']
    dataset_test = pd.read_csv(test_file_name)
    #abaondon this column since > 50% NA
    dataset = dataset.drop(columns = ['ID', 'months_since_last_delinq', 'race', 'loan_paid'])
    dataset_test = dataset_test.drop(columns = ['ID', 'months_since_last_delinq', 'race', 'loan_paid'])
    
    numerical = ['requested_amnt', 'annual_income', 'employment_length', 'debt_to_income_ratio', 'fico_score_range_low', 'fico_score_range_high', 'revolving_balance', 'total_revolving_limit', 'public_bankruptcies']
    #categorical = ['loan_duration', 'home_ownership_status', 'reason_for_loan', 'extended_reason', 'employment_verified',  'state', 'fico_inquired_last_6mths', 'any_tax_liens']
    categorical = ['loan_duration', 'home_ownership_status', 'reason_for_loan', 'extended_reason', 'employment_verified','state', 'fico_inquired_last_6mths', 'any_tax_liens', 'employment', 'zipcode']
    
    
    def _check_lien():
        dataset_cp['any_tax_liens'].fillna('NA', inplace=True)
        unique = dataset['any_tax_liens'].unique()
        result = {}
        for uni in unique:
            result[uni] = [0, 0]
        for i in range(dataset.shape[0]):
            result[dataset_cp['any_tax_liens'][i]][int(dataset_cp['loan_paid'][i])] += 1
        print(result)
        for uni in unique:
            result[uni] = np.array(result[uni]) / sum(np.array((result[uni])))
        print(result)
        
        
      
    def check_data():
        for column in dataset.columns:
            print(column + ": " + str((dataset[column].isnull().sum() / dataset.shape[0])))
            print(column + ": " + str(len(dataset[column].unique())))

    #clean data
    def _clean():
        dataset[['total_revolving_limit']] = dataset[['total_revolving_limit']] / 100
        dataset[['total_revolving_limit']] = dataset[['total_revolving_limit']] / 100 
        dataset_test[['total_revolving_limit']] = dataset_test[['total_revolving_limit']] / 100
        dataset_test[['total_revolving_limit']] = dataset_test[['total_revolving_limit']] / 100  
        
    # Label Encoding
    def _label_encoding():
        for cat in categorical:
            dataset[cat].fillna('NA', inplace=True)
            dataset_test[cat].fillna('NA', inplace=True)
        for cat in numerical:
            dataset[cat].fillna(dataset[cat].mean(), inplace=True)
            dataset_test[cat].fillna(dataset[cat].mean(), inplace=True)
        
    #change to others
    def _clean_2():
        clean_cat = ['extended_reason', 'state', 'home_ownership_status', 'zipcode', 'employment']
        #clean_cat = ['extended_reason', 'state', 'home_ownership_status']
        for cat in clean_cat:
            counts = Counter()
            for count in dataset[cat]:
                counts[count] += 1
            most_common = []
            if cat == 'extended_reason':
                most_common = [x[0] for x in counts.most_common(8)]
            elif cat == 'state':
                most_common = [x[0] for x in counts.most_common(20)]
            elif cat == 'home_ownership_status':
                most_common = [x[0] for x in counts.most_common(3)]
            elif cat == 'zipcode':
                most_common = [x[0] for x in counts.most_common(30)]
            elif cat == 'employment':
                most_common = [x[0] for x in counts.most_common(30)]
            '''    
            sum = 0
            if cat == 'zipcode' or cat == 'employment':
                print(cat)
                for common in most_common:
                    print(counts[common] / dataset.shape[0])
                    sum += counts[common] / dataset.shape[0]
                print(sum)
            '''
            dataset[cat] = dataset[cat].apply(lambda i: i if i in most_common else 'Others')
            dataset_test[cat] = dataset_test[cat].apply(lambda i: i if i in most_common else 'Others')
        most_common = [0.0]
        dataset['any_tax_liens'] = dataset['any_tax_liens'].apply(lambda i: i if i in most_common else 1.0)
        dataset_test['any_tax_liens'] = dataset_test['any_tax_liens'].apply(lambda i: i if i in most_common else 1.0)
        dataset[['any_tax_liens']] = dataset[['any_tax_liens']].astype(str)
        dataset_test[['any_tax_liens']] = dataset_test[['any_tax_liens']].astype(str)
        
        dataset['fico_inquired_last_6mths'] = dataset['fico_inquired_last_6mths'].apply(lambda i: i if i in most_common else 1.0)
        dataset_test['fico_inquired_last_6mths'] = dataset_test['fico_inquired_last_6mths'].apply(lambda i: i if i in most_common else 1.0)
        dataset[['fico_inquired_last_6mths']] = dataset[['fico_inquired_last_6mths']].astype(str)
        dataset_test[['fico_inquired_last_6mths']] = dataset_test[['fico_inquired_last_6mths']].astype(str)
       
        
    def _getEmployeement():
        dataset.loc[dataset['employment_length'].str.find("<") >= 0, 'employment_length'] = 0
        dataset_test.loc[dataset_test['employment_length'].str.find("<") >= 0, 'employment_length'] = 0
        dataset.loc[dataset['employment_length'].str.find("+") > 0, 'employment_length'] = 10
        dataset_test.loc[dataset_test['employment_length'].str.find("+") > 0, 'employment_length'] = 10

        dataset.loc[dataset['employment_length'].str.find("1") >= 0, 'employment_length'] = 1
        dataset_test.loc[dataset_test['employment_length'].str.find("1") >= 0, 'employment_length'] = 1
        dataset.loc[dataset['employment_length'].str.find("2") >= 0, 'employment_length'] = 2
        dataset_test.loc[dataset_test['employment_length'].str.find("2") >= 0, 'employment_length'] = 2
        dataset.loc[dataset['employment_length'].str.find("3") >= 0, 'employment_length'] = 3
        dataset_test.loc[dataset_test['employment_length'].str.find("3") >= 0, 'employment_length'] = 3
        dataset.loc[dataset['employment_length'].str.find("4") >= 0, 'employment_length'] = 4
        dataset_test.loc[dataset_test['employment_length'].str.find("4") >= 0, 'employment_length'] = 4
        dataset.loc[dataset['employment_length'].str.find("5") >= 0, 'employment_length'] = 5
        dataset_test.loc[dataset_test['employment_length'].str.find("5") >= 0, 'employment_length'] = 5
        dataset.loc[dataset['employment_length'].str.find("6") >= 0, 'employment_length'] = 6
        dataset_test.loc[dataset_test['employment_length'].str.find("6") >= 0, 'employment_length'] = 6
        dataset.loc[dataset['employment_length'].str.find("7") >= 0, 'employment_length'] = 7
        dataset_test.loc[dataset_test['employment_length'].str.find("7") >= 0, 'employment_length'] = 7
        dataset.loc[dataset['employment_length'].str.find("8") >= 0, 'employment_length'] = 8
        dataset_test.loc[dataset_test['employment_length'].str.find("8") >= 0, 'employment_length'] = 8
        dataset.loc[dataset['employment_length'].str.find("9") >= 0, 'employment_length'] = 9
        dataset_test.loc[dataset_test['employment_length'].str.find("9") >= 0, 'employment_length'] = 9        
        
    
    def check_data_2(final_test):
        for column in final_test.columns:
            if final_test[column].isnull().sum() > 0:
                print(column + ": " + str((final_test[column].isnull().sum())))
                print(column + ": " + str(len(final_test[column].unique())))

    def compare_dist(tr, te):
        for column in tr.columns:
            sum_tr = 0
            sum_te = 0
            counts_tr = Counter()
            for count in tr[column]:
                counts_tr[count] += 1
            
            counts_te = Counter()
            for count in te[column]:
                counts_te[count] += 1
            
            counts_t = counts_tr.most_common(20)
            counts_e = counts_te.most_common(20)
        exit(0)
            
            
        
 
    
    def compare_fico(tr, te):
        '''
        counts_tr = Counter()
        for count in tr['fico_score_range_high']:
            counts_tr[count] += 1
        
        counts_te = Counter()
        for count in te['fico_score_range_high']:
            counts_te[count] += 1
        
        counts_tr = sorted(counts_tr.items())
        counts_te = sorted(counts_te.items())
        '''
        sum = 0
        counts_tr = [(654, 130787), (659, 1397), (664, 18602), (669, 17951), (674, 17844)
                     , (679, 16067), (684, 15959), (689, 13440), (694, 13228), (699, 11846)
                     , (704, 10916), (709, 9840), (714, 8428), (719, 7491), (724, 6497)
                     , (729, 5257), (734, 4398), (739, 3471), (744, 3119), (749, 2609) 
                     , (750, 20), (754, 2292), (759, 1974), (764, 1655), (769, 1520)
                     , (774, 1346), (779, 1150), (784, 1059), (789, 828), (794, 745), (799, 597)
                     , (804, 554), (809, 485), (814, 357), (819, 286), (824, 185), (829, 140)
                     , (834, 108), (839, 56), (844, 39), (850, 25)]               
        tr_li = []
        #te_li = []
        for i in counts_tr:
            sum += i[1]
        for i in counts_tr:
            tr_li.append((i[0], i[1] / sum))
        
        #for j in counts_te:
        #    te_li.append((j[0], j[1] / te.shape[0]))
            
        print(tr_li)
        print(counts_tr)
        
    
    def force_fico_dist(dataset, y):
        print(y.value_counts()[1])
        print(y.value_counts()[0])
        target = [(654, 130787), (659, 1397), (664, 18602), (669, 17951), (674, 17844)
                     , (679, 16067), (684, 15959), (689, 13440), (694, 13228), (699, 11846)
                     , (704, 10916), (709, 9840), (714, 8428), (719, 7491), (724, 6497)
                     , (729, 5257), (734, 4398), (739, 3471), (744, 3119), (749, 2609) 
                     , (750, 20), (754, 2292), (759, 1974), (764, 1655), (769, 1520)
                     , (774, 1346), (779, 1150), (784, 1059), (789, 828), (794, 745), (799, 597)
                     , (804, 554), (809, 485), (814, 357), (819, 286), (824, 185), (829, 140)
                     , (834, 108), (839, 56), (844, 39), (850, 25)]     
        
        idx = []
        sum = 0
        for val, num in target:
            cur_idx = dataset[(dataset['fico_score_range_high'] == val)].index.tolist()
            if len(cur_idx) != num:
                cur_idx = random.sample(cur_idx, num)
            idx.extend(cur_idx)
            sum += num
        idx = sorted(idx)
        dataset = dataset.iloc[idx, :]
        y = y.iloc[idx]
        print(y)
        print(y.value_counts()[1])
        print(y.value_counts()[0])
        return dataset, y
        
    def force_dist(dataset, y):
        print(y.value_counts()[1])
        print(y.value_counts()[0])
        idx = []
        sum = 0
        #235447
        #400260
        for val, num in [(0,200130), (1,430000)]:
            cur_idx = y[(y == val)].index.tolist()
            if len(cur_idx) != num:
                cur_idx = random.sample(cur_idx, num)
            idx.extend(cur_idx)
            sum += num
        idx = sorted(idx)
        dataset = dataset.iloc[idx, :]
        y = y[idx]
        print(y)
        print(y.value_counts()[1])
        print(y.value_counts()[0])
        return dataset, y

        
        
        
        
    #_check_data()
    #check_data()
    #_clean()
    #_normalize()
    #_check_lien()
    #
    print(y.value_counts()[1])
    print(y.value_counts()[0])
    _getEmployeement()
    
    #print(dataset['fico_score_range_high'].describe())
    #print(y.value_counts()[1])
    #print(y.value_counts()[0])
    #dataset, y = force_dist(dataset, y)
    _label_encoding()
    _clean_2()
    
    val_X, val_y = force_fico_dist(dataset, y)
    dataset, y = force_dist(dataset, y)
    #print(dataset['fico_score_range_high'].describe())
    #print(dataset_test['fico_score_range_high'].describe())
    #compare_dist(dataset, dataset_test)
    #print(dataset['debt_to_income_ratio'].describe())
    #print(dataset_test['debt_to_income_ratio'].describe())
    
    
    #check_data()
    
    one_hot_encoded_training_predictors = pd.get_dummies(dataset)
    one_hot_encoded_test_predictors = pd.get_dummies(dataset_test)
    one_hot_encoded_val_predictors = pd.get_dummies(val_X)
    
    final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                        join='left', 
                                                                        axis=1)
    final_train, final_val = one_hot_encoded_training_predictors.align(one_hot_encoded_val_predictors,
                                                                        join='left', 
                                                                        axis=1)
    

    final_train.columns = final_train.columns.str.strip()
    final_test.columns = final_train.columns.str.strip()
    final_val.columns = final_val.columns.str.strip()
        
    #sclar
    def _scalar(train, test, val):
        scaler = StandardScaler()
        train[numerical]= scaler.fit_transform(train[numerical])
        test[numerical] = scaler.transform(test[numerical])
        val[numerical] = scaler.transform(val[numerical])
        return train ,test, val
    
    final_test, final_test, final_val = _scalar(final_train, final_test, final_val)
    
    print(final_test['fico_score_range_high'].describe())
    print(final_val['fico_score_range_high'].describe())
        
    #check_data_2(final_test)
    #compare_fico(final_train, final_test)
    #exit(0)
    
    '''
    print(final_train)
    print(final_test)

    pca = PCA(n_components=20)
    pca.fit(final_train)
    np.set_printoptions(threshold= sys.maxsize)
    print(final_train.shape)
    print(pca.feature_names_in_)
    print(pca.explained_variance_ratio_.shape)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    '''
    
    
    return final_train, y, final_test, final_val, val_y


        
