import s3fs
import boto3
import io
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from math import sqrt
from numpy import concatenate
from numpy import array
from pandas import DataFrame
from pandas import concat

import time
from datetime import datetime
from datetime import date
import dateutil

import itertools
from collections import Counter
from subprocess import check_output

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import svm,tree

from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, BatchNormalization

from datetime import datetime, timedelta
from dateutil.relativedelta import *
import lightgbm as lgb
import numpy
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input
from keras.callbacks import EarlyStopping
import math
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

def il(pkg):
    """
    Input:
        pkg - Package name you want to import
    
    Output:
        Install(If package is not installed already) and Import the package
    """
    import importlib
    import subprocess
    try:
        return importlib.import_module('%s'%(pkg))
    except:
        subprocess.call(['pip','install','--user',pkg])
        return importlib.import_module('%s'%(pkg))


def il_all(pkg_lst=[]):
    """
    Input:
        pkg_lst - A list of packages you want to import, if passed blank only the default packages will be imported
        
    Output:
        A list of all packages imported
    """
    pkg_lst_default = ['pandas','numpy','re','os','subprocess','gc','seaborn','matplotlib.pyplot','jellyfish','tensorflow','keras',
                   'sys','pickle','s3fs','dask','dask.dataframe','collections']
    for l in pkg_lst:
        pkg_lst_default.append(l)
    return [il(j) for j in pkg_lst_default]


def folder_structure_from_s3(path,endswith_='zip'):
        """
        Input: 
            path - s3 folder path
            endswith_ - files endwith what structure

        Output: list of zip files in the given s3 folder path
        """
        import subprocess
        l = subprocess.check_output(['aws','s3','ls',path])
        ll = str(l).split('\\n')
        sub_folder_list = [i.split(' ')[-1] for i in ll]
        sub_folder_list = [i for i in sub_folder_list if i.endswith(endswith_)]
        return sub_folder_list
    

def unzip_s3zips(s3_path,s3_upload=True):
    """
    A function to Unzip and upload all zipped files located in a s3 folder
    Input: 
        s3_path - str, s3 folder path
        s3_upload - Boolean (True/False), if True files will be uploaded after unzipping
    
    Output: Zipped files unzipped and uploaded to s3
    """
    import os
    import subprocess
    
    path = s3_path

    zip_files = folder_structure_from_s3(path) #list of zip files in the path

    for f in zip_files:
        fold = f.split('.')[0] #getting the file names without .zip extension
        os.mkdir(fold)         #making a temporary directory with the file name
        subprocess.call(['aws','s3','cp',path+f,fold+'/.'])  #copying the corresponding file from s3 in the folder
        os.chdir(fold)    #going inside the folder
        subprocess.call(['unzip',f]) #unzipping the file contents in the folder
        subprocess.call(['rm',f]) #removing the zip file from the folder
        os.chdir('..')  #going back outside the folder
        
        if s3_upload:
            subprocess.call(['aws','s3','cp',fold,path+fold,'--recursive']) #pushing the folder contents into corresponding s3 folder
            subprocess.call(['rm','-rf',fold])  #deleting the folder from instance
        else:
            continue
    

def read_table(path_,sep=None):
    """
    A single Function to read data from s3/local drive, in any of the formats, like csv, parquet, excel.
    
    Input:
        path_ - s3 file path/local drive file path. s3 file path can be passed without the 's3://' suffix, the function will automatically detect the path type and read the file accordingly
        sep - If None passed, the function will detect the separator automatically. 
        
    Output: Pandas Dataframe
    """
    if path_.endswith('.csv'):
        func_ = pd.read_csv
    elif path_.endswith('.parquet'):
        func_ = pd.read_parquet
    elif path_.endswith('.xlsx'):
        func_ = pd.read_excel
    else:
        func_ = pd.read_csv
    
    if sep == None:
        for sep in [',','|','\t','\n',';']:
            try:
                dat = func_(path_,sep)
                break
            except:
                try:
                    dat = func_('s3://'+path_,sep)
                    break
                except:
                    try:
                        dat = func_(path_,sep)
                        break
                    except:
                        try:
                            dat = func_('s3://'+path_)
                            break
                        except:
                            pass
                        
    else:
        try:
            dat = func_(path_,sep)
        except:
            dat = func_('s3://'+path_,sep)
        
    return dat



def change_types(data,int_cols=[],date_cols=[],float_cols=[]):
    """
    Change format of different columns of a Dataframe
    Input: 1) DataFrame 2) Integer Column List 3) Date Column List 4) Float Column List
    
    Output: Dataframe with Change Data Formats
    """
    for col in int_cols:
        data[col] = data[col].astype('int')
    for col in date_cols:
        data[col] = pd.to_datetime(data[col])
    for col in float_cols:
        data[col] = data[col].astype('float')
    return data


def read_csv(f_path, delimiter):
    """
    Read Spark DataFrame
    """
    return spark.read.format('com.databricks.spark.csv') \
            .option('header','True') \
            .option('delimiter', delimiter) \
            .load(f_path)


def zipindexdf(df):
    """
    Adding Index Column to a Spark Dataframe
    """
    schema_new = df.schema.add("idx", LongType(), False)
    return df.rdd.zipWithIndex().map(lambda l: list(l[0]) + [l[1]]).toDF(schema_new)


def extract_number_from_string(x):
    '''Function to extract number from string'''
    import re
    try:
        s=re.findall(r'\d+', x)
        s = '_'.join(s)
    except:
        s=-999
    return s


def take_out_recent(df,id_col,date_col):
    """
    To Take out only the recent row corresponding to a primary key in a passed dataframe
    """
    df_id = df[[id_col,date_col]].drop_duplicates()

    df_id = df_id.sort_values(by=[id_col,date_col],ascending=False)

    df_id = df_id.drop_duplicates(subset=id_col,keep='first')

    df_mod = pd.merge(df,df_id,on=[id_col,date_col],how='inner')
    return df_mod


def spark_to_pandas_df(part_files_path,delim=','):
    """
    To Make a single Pandas dataframe from Spark Saved Part DataFiles
    Input:
        a) Spark saved part files path
        b) Delimiter
    
    Outpt:
        Pandas Dataframe
    """
    import dask.dataframe as dd
    import csv
    import numpy as np
    
    df = dd.read_csv(part_files_path+'*',dtype='object',delimiter=delim,quoting=csv.QUOTE_NONE, encoding='utf-8')
    df_p = df.compute()
    df_p = df_p.replace('""',np.nan)
    return df_p


def uniform_dtype(x):
    """
    Uniform Data Type accross entire Dataframe
    Input:
        1) x - DataFrame each Element
    Output:
        Data Type Changed element
    """
    try:
        return float(x)
    except:
        if type(x)=='str': 
            return x.upper().strip()
        else:
            return x

        
def date_selection(x,date_column,avg_days_to_select):
    """
    A groupby function, applicable on Pandas Groupby dataframe, which can be apply for each of the groupby IDs and for each ID 
    one date will be selected, calculated from the first date and in a difference of avg_days_to_select from First selected
    date.
    
    Input: 
        1) x - Groupedby pandas Dataframe
        2) date_column - On which Date Column the operation should be done
        3) avg_days_to_select - Calculated Number of Days to select the date from the First Date for an ID
    
    Output:
        Selected Date for the ID
    """
    import numpy as np
    from datetime import timedelta
    first_date = x[date_column].min()
    cutoff_calculated_date = np.datetime64(first_date + timedelta(avg_days_to_select))
    all_dates = list(x[date_column].unique())
    all_day_diff = [np.abs((i-cutoff_calculated_date)/np.timedelta64(1, 'D')) for i in all_dates]
    return all_dates[np.argmin(all_day_diff)]


def find_match(var,lis,threshold_score=0.8):
    """
    Find Highest Matched String Element from a passed list for a given String
    
    Input:
        1) var - Passed String
        2) lis - List of Strings
        3) threshold_score - Threshold Max Score Allowed
    
    Output:
        Highest Matched String from Lis
    """
    import jellyfish as jf
    temp_lis = []
    for var_ in lis:
        score = jf.jaro_winkler( var.lower(), var_.lower())
        temp_lis.append((var_,score))
    temp_lis.sort(key=lambda x: x[1])
    try:
        scor = temp_lis[-1][1]
        if scor >= threshold_score:
            return temp_lis[-1][0]
        else:
            return 'NA'
    except:
        return 'NA'
    
    
def find_name_w_rearrange(s1,s2):
    try:
        collect = []
        for i in range(len(s1)):
            sub_s = s1[0:i+1]
            if sub_s in s2:
                collect.append(sub_s)
            else:
                break
        name_part = collect[-1]
        other_name_part = [j for j in s1.split(name_part) if len(j) != 0][0]

        combis = [name_part+other_name_part , other_name_part+name_part]
    except:
        combis = []
    return combis


def compare_plot_distribution(data,dist_variable,comparison_variable,bins=20,kde=True,title=None,xlabel=None,ylabel=None):
    """
    A plotting function, which can plot histogram(Distribution) of a continuous passed variable, for different categories of a
    second passed variable
    
    Input:
        1) data - Dataframe
        2) dist_variable - Distribution/ Continuous Variable for which we want to plot the histograms
        3) comparison_variable - The second categorical variable on which we will compare the histograms
        4) bins - Number of bins of the histograms
        5) kde - Kernel Density Estimation, will be plotted if passed as True
        6) title - Title of the Plot
        7) xlabel - Xlabel for the X-Axis
        8) ylabel - Ylabel for the Y-Axis
    Output:
        Comparison Plot of Multiple Histograms
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    cats = list(data[comparison_variable].unique())
    plt.figure(figsize=(15,8))
    
    for cat in cats:
        sns.distplot(data[data[comparison_variable]==cat][dist_variable],bins=bins,kde=kde)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, data[dist_variable].max()])
    plt.legend(cats)
    

def plot_multiples_same_layout(data,cols,plot_in_nrows=2,xlabel=None,ylabel=None,plot_type='hist'):
    """
    A function to plot same type of statistical plottings for multiple variables in a same layout
    
    Input:
        data - the dataframe
        cols - list, list of columns we want to plot for
        plot_in_rows - int, In the picture layout, in how many rows do we want to plot the garphs. Default value passed is 2
        xlabel - str, Label of the X axis, default is None
        ylabel - str, Label of the Y axis, default is None
        plot_type - str, Either 'hist' or 'box' can be passed as keyword, 'hist' is for histogram and 'box' is for boxplot   
    
    Output: Graphs plotted in a single layout
        
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    n_cols = len(cols)
    plt.figure(figsize=(20,8))
    for i, column in enumerate(cols):
        plt.subplot(plot_in_nrows,n_cols/plot_in_nrows,i+1)
        if plot_type == 'hist':
            sns.distplot(data[column],kde=False)
        elif plot_type == 'box':
            sns.boxplot(data[column],orient='v')
        else:
            raise TypeError('Please pass a valid plot type')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(column)

        
def create_dataframe_dic(root_path,table_list=None,file_formats='.txt',sep=','):
    """
    Create a DataFrame Dictionary with table name as keys and the dask table as values of the dictionary
    
    Input:
        root_path - Root s3 path where the files are stored
        table_list - No table list passed if None values passed, otherwise specific table names which we want to load
        can be passed in list format
        file_formats - the format of the files stored in the s3 root path, like txt, csv etc.
        
    Output:
        A Dictionary of dask Dataframe objects
    """
    import dask.dataframe as pd 
    dataframe_dic = {}
    if table_list==None:
        table_list = folder_structure_from_s3(root_path,file_formats)
        table_list = [j.split(file_formats)[0] for j in table_list]
    for tab in table_list:
        try:
            dataframe_dic[tab] = pd.read_csv(root_path+tab+file_formats,sep=sep,low_memory=False,dtype='object')
            dataframe_dic[tab] = dataframe_dic[tab].drop_duplicates()
            dataframe_dic[tab].columns = map(str.lower, dataframe_dic[tab].columns)
        except Exception as e:
            print(e)
            dataframe_dic[tab] = ''
    return dataframe_dic


def run_auto_join(path_join_schema_in_table_format,root_path_of_tables,master_table):
    """
    A function to perform auto join accross different tables, as per the joining schema provided
    
    Input:
        path_join_schema_in_table_format - File path of the join schema, in csv dataframe format.
        root_path_of_tables - The s3 file path where all the tables are stored
        master_table - str, The name of the master table
        
    Output: A Joined Dataframe
    """
    import pandas as dd
    import csv
    import sys
    import subprocess
    import os
    csv.field_size_limit(sys.maxsize)

    import copy
    import pickle
    import gc
    import s3fs

    import dask
    from dask.diagnostics import ProgressBar
    from dask.distributed import Client, LocalCluster, progress
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    import collections
    import dask.dataframe as pd
    
    map_ = pd.read_csv(path_join_schema_in_table_format)
    all_tables = list(set(map_['Record/Table'].unique().compute()).union(set(map_['Referenced Record/Table'].unique().compute())))
    map_['Column Name'] = map_['Column Name'].apply(lambda x: x.lower())
    map_['Reference Column '] = map_['Reference Column '].apply(lambda x: x.lower())
    
    dataframe_dic = create_dataframe_dic(root_path_of_tables)
    lis_ = map_['Referenced Record/Table'].unique().compute()
    
    def find_child(x):
        return list(map_['Record/Table'][map_['Referenced Record/Table']==x].values.compute())
    
    def existence_checking(p):
        try:
            temp[p]
            y = True
        except:
            y = False
        return y

    def check(p1,p2):
        try:
            graph[p2][p1]
            y = True
        except:
            y = False
        return y
    
    temp = {}
    for token in map_['Record/Table'].values.compute():
        if not existence_checking(token):
            temp[token] = list(map_['Referenced Record/Table'][map_['Record/Table']==token].unique().compute())[0]
        else:
            temp[token] = list(map_['Referenced Record/Table'][map_['Record/Table']==token].unique().compute())[0]
    
    reverse_dic = {}
    keys = list(set([k for k,v in temp.items()]))
    values = list(set([v for k,v in temp.items()]))

    graph = {}
    for k in values:
        temp_ = {}
        for v in keys:
            if v != k:
                if temp[v] == k:
                    temp_[v] = {}
        graph[k] = temp_

    for k,v in graph.items():
        for k1, v1 in graph.items():
            if check(k,k1):
                graph[k1][k] = graph[k]
                
    graph_ = {}
    graph_[master_table] = graph[master_table]
    
    def findKeys(data, key,result):    
        if key in list(data.keys()):
            result.append(data[key])
        else:
            for i in list(data.keys()):
                if type(data[i]) == dict:
                    findKeys(data[i],key,result)
        return result
    
    def under_leafs(elem):
        l = findKeys(graph_,elem,[])
        childrens = list(l[0].keys())
        return childrens
    
    
    i =1
    lis = []
    prime_element = [master_table]
    while i >0:
        children = []
        for p in prime_element:
            children.extend(under_leafs(p))
        for c in children:
            temp = {}
            temp[c] = i
            lis.append(temp)
        prime_element = children
        i+=1
        if i > 49:
            break
            
    extended_dic = {}
    for i in lis:
        for k,v in i.items():
            extended_dic[k] = v
            
    extended_dic[master_table] = 0
    
    def find_elem_at_dist(dist):
        all_keys = [k for k,v in extended_dic.items() if v==dist]
        return all_keys
    
    depth = max(extended_dic.items(), key=lambda x: x[1])[1]
    
    i=1
    while i>=1:
        max_val = max(extended_dic.items(), key=lambda x: x[1])[1]
        max_all_keys = [k for k,v in extended_dic.items() if v==max_val]

        temp = {}
        max_all_keys_ = [i.split('/')[0] for i in max_all_keys]
        for l in range(len(max_all_keys)):
            dist = max_val-1
            all_elms = find_elem_at_dist(dist)
            for k1 in all_elms:
                if max_all_keys_[l] in under_leafs(k1):
                    temp[k1+'/'+max_all_keys[l]] = dist

        temp_dic = copy.deepcopy(extended_dic)

        for k1,v1 in temp.items():
            k2 = k1.split('/')
            for k,v in extended_dic.items():
                if k not in max_all_keys:
                    if k in k2:
                        itm = temp_dic[k]
                        del temp_dic[k]
                else:
                    del temp_dic[k]
            temp_dic[k1] = itm
            extended_dic = copy.deepcopy(temp_dic)

        temp1_dic = copy.deepcopy(extended_dic)

        for k,v in temp1_dic.items():
            tabs = k.split('/')
            if len(tabs) == 2:
                if tabs[1] in ['Endnote','OverrideEndnote']:
                    key1 = list(set(map_['Reference Column '][(map_['Referenced Record/Table']==tabs[1])&(map_['Record/Table']==tabs[0])].values.compute()))
                    key2 = list(set(map_['Column Name'][(map_['Referenced Record/Table']==tabs[1])&(map_['Record/Table']==tabs[0])].values.compute()))

                    key1 = [i.strip() for i in key1[0].split(',')]
                    key2 = [i.strip() for i in key2[0].split(',')]
                else:
                    key1 = list(set(map_['Reference Column '][(map_['Referenced Record/Table']==tabs[0])&(map_['Record/Table']==tabs[1])].values.compute()))
                    key2 = list(set(map_['Column Name'][(map_['Referenced Record/Table']==tabs[0])&(map_['Record/Table']==tabs[1])].values.compute()))

                    key1 = [i.strip() for i in key1[0].split(',')]
                    key2 = [i.strip() for i in key2[0].split(',')]

                try:
                    common_columns = [i for i in dataframe_dic[tabs[0]].columns if i in dataframe_dic[tabs[1]].columns]
                    col_dic = {}
                    for col in common_columns:
                        if col not in key1+key2:
                            col_dic[col] = col+'_'+tabs[1]

                    dataframe_dic[tabs[1]] = dataframe_dic[tabs[1]].rename(columns=col_dic)
                except Exception as e:
                    print(e)


                try:
                    print('%s with %s'%(tabs[0],tabs[1]))

                    print(len(dataframe_dic[tabs[0]].columns))#,dataframe_dic[tabs[0]].index.size.compute())
                    try:
                        dataframe_dic[tabs[0]] = pd.merge(dataframe_dic[tabs[0]],dataframe_dic[tabs[1]],left_on=key2,right_on=key1,how='left')
                    except:
                        try:
                            dataframe_dic[tabs[0]] = pd.merge(dataframe_dic[tabs[0]],dataframe_dic[tabs[1]],left_on=key1,right_on=key2,how='left')
                        except:
                            pass

                    print(len(dataframe_dic[tabs[0]].columns))#,dataframe_dic[tabs[0]].index.size.compute())
                    del extended_dic[k]
                    extended_dic[tabs[0]] = v
                except Exception as e:
                    print(k,e)
                    del extended_dic[k]
                    pass
            else:
                pass

            gc.collect()

        i+=1
        if i>depth+1:
            break
            
    return dataframe_dic[master_table]


def make_data_report(data_description_file,s3_data_path,sep='|',type_col={'Type':''},length_col={'Prec':'Length'}):
    """
    A function to automatically create data reports
    
    Input:
        data_description_file - An excel file with multiple tabs with different table names, containing variable informations like dtype, value length etc, for each of the tables
        s3_data_path - path of s3 data location
        sep - File field seperator/ delimiter
        type_col - The column name of the information containing the variable types for each column of each table in the data_description_file. Value passed should be in dictionary format, where key will be the column name and value will be the alternative column name(incase the key column is blank value column will be used as type_col). If dont have any alternative column value should be passed as empty string ('').
        length_col - The column name of the information containing the variable value length for each column of each table in the data_description_file. Value passed should be in dictionary format, where key will be the column name and value will be the alternative column name(incase the key column is blank value column will be used as length_col). If dont have any alternative column value should be passed as empty string ('').
        
    Output:
        A dictionary, having table names as keys and values are boolean (TRUE/FALSE). TRUE indicates all the columns have been tested and passed successfully in terms of dtype and length and FALSE indicates there is some issue with any of the variable of the table in terms of either dtype or length. The detailed reports will be saved in a folder (automatically created if doesnt exist), named 'report', for each of the table. 
    
    """
    
    from essentials import essentials
    libs = il_all()
    description = libs[0].ExcelFile(data_description_file)    
    sheet_names = description.sheet_names
    
    dtype_map = libs[0].read_csv('s3://__data_path__/sql_python_dtype.txt',sep='\t')
    
    dtype_dict = dict(zip(dtype_map.SQLtype, dtype_map.Python_type))
    
    #check Data Types
    def check_type(col,data,desc_data,F):
        desc_data = desc_data
        type_sql = list(desc_data[list(type_col.keys())[0]][desc_data['Column_name']==col].values)[0]
        
        if len(str(type_sql).strip()) == 0:
            type_sql = list(desc_data[type_col[list(type_col.keys())[0]]][desc_data['Column_name']==col].values)[0]
        type_python = dtype_dict[type_sql]
        try:
            data[col].astype(type_python)
            r = '%s dtype is verified and it is of %s type'%(col,type_python)
            F.write(r+'\n')
            print(r)
            y = True
        except:
            r = '>>> %s is not of %s type'%(col,type_python)
            F.write(r+'\n')
            print(r)
            y = False
        return y
    
    
    #check the Max length of a variable values
    def check_length(col,data,desc_data,F):
        desc_data = desc_data
        max_length = list(desc_data[list(length_col.keys())[0]][desc_data['Column_name']==col].values)[0]
        if len(str(max_length).strip()) == 0:
            max_length = list(desc_data[length_col[list(length_col.keys())[0]]][desc_data['Column_name']==col].values)[0]

        temp_x = data[col].astype('str')
        try:
            temp_x = temp_x.apply(lambda x: ''.join(x.split('.')))
        except:
            pass
        len_temp_x = temp_x.apply(lambda x:len(x))
        max_ = len_temp_x.max()
        if type(max_length) == str:
            y = True
        else:
            if max_ <= max_length:
                r = '%s length is verified and it is of max length %d'%(col,max_)
                F.write(r+'\n')
                print(r)
                y = True
            else:
                r = '*** %s length max %d is not matching with specified length %d'%(col,max_,max_length)
                F.write(r+'\n')
                print(r)
                y = False
            return y 
        
    def validate_col(col,data,desc_data,F):
        F.write('\n----------------------\n')
        if check_type(col,data,desc_data,F) and check_length(col,data,desc_data,F):
            return True
        else:
            return False
        
    def validate_data(data,desc_data,F):
        cols = data.columns
        check_list = [validate_col(col,data,desc_data,F) for col in cols]
        print(check_list)
        if all(check_list):
            return True
        else:
            return False
    
    files = essentials.folder_structure_from_s3(s3_data_path,'.txt')
    def run_validation(key):
        data = libs[0].read_csv(s3_data_path+key+'.txt',sep=sep,encoding='latin')
        if not libs[3].path.exists('report/'):
            libs[3].mkdir('report/')
        F = open('report/'+key+'.txt','w')
        try:
            return validate_data(data,description.parse(key),F)
        except Exception as e:
            print(e)
            pass
        
    validation_result = {}
    for file in files:
        key = file.split('.')[0]
        print(key)
        try:
            validation_result[key] = run_validation(key)
        except:
            validation_result[key] = 'No Data'
    
    return validation_result


# Wanli
def _write_dataframe_to_csv_on_s3(dataframe, filename, s3_destination_bucket):
    """ Write a dataframe to a CSV on S3 """
    import boto3
    import sys
    if sys.version[0] == '2':
        from io import BytesIO as StringIO  # for python 2.7
    else:
        from io import StringIO

    print("Writing {} records to {}".format(len(dataframe), filename))
    # Create buffer
    csv_buffer = StringIO()
    # Write dataframe to buffer
    dataframe.to_csv(csv_buffer, sep="|", index=False)
    # Create S3 object
    s3_resource = boto3.resource("s3")
    # Write buffer to S3 object
    s3_resource.Object(s3_destination_bucket, filename).put(Body=csv_buffer.getvalue())


def is_numeric(string):
    """
    Check if a string could transfer to number

    :param string: A string value
    :return: True/False
    """
    import re
    return bool(re.match("[+-]?[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?", string))


def hyper_test_pvalue(label, feature, dropna=True):
    """
    Hypothesis Testing P-value for the target label with a given feature

    :param label: target variable
    :param feature: the variable against which you will be testing the target
    :param dropna: If True, Nan values will be dropped
    :return: P-value, float
    """
    from sklearn import preprocessing
    from sklearn.feature_selection import chi2, f_classif
    import numpy as np
    import pandas as pd

    df = pd.DataFrame()
    df['label'] = label.apply(lambda x: 'label' + str(x))
    df['feature'] = feature.values
    if dropna:
        no_nan = df.dropna().copy()
        # print('nan rate for %s is %.2f'%('feature',1-no_nan.shape[0]/df.shape[0]))
    else:
        # print("replace NAN with string 'NAN'")
        df.loc[:, 'feature'] = df.loc[:, 'feature'].replace(np.nan, 'NAN', regex=True)
        df.loc[:, 'label'] = df.loc[:, 'label'].replace(np.nan, 'NAN', regex=True)
        no_nan = df.copy()
    if len(no_nan) < 20:
        return np.nan
    if 'int' not in str(no_nan['label'].dtype):
        le_label = preprocessing.LabelEncoder()
        temp_list = list(map(lambda x: str(x), no_nan['label'].unique()))
        le_label.fit(temp_list)
        no_nan.loc[:, 'label'] = le_label.transform(no_nan['label'])
    if str(feature.dtype) in ('category', 'objective', 'bool') or feature.nunique() < 0.05 * len(feature):
        le = preprocessing.LabelEncoder()
        no_nan['feature'] = list(map(lambda x: str(x), no_nan['feature']))
        le.fit(no_nan['feature'].unique())
        no_nan.loc[:, 'feature'] = le.transform(no_nan.loc[:, 'feature'])

        (score, p_value) = chi2(no_nan['feature'].values.reshape(-1, 1), no_nan['label'])
    else:
        (score, p_value) = f_classif(no_nan['feature'].values.reshape(-1, 1), no_nan.loc[:, 'label'])
    return p_value[0]


def heat_map_p_value(raw_df, category_features, features):
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    df_matrix = pd.DataFrame(index=features)
    matrix = -1 * np.ones([len(category_features), len(features)])
    for i, label in enumerate(category_features):
        for j, feature in enumerate(features):
            p_value = hyper_test_pvalue(raw_df[label], raw_df[feature])
            matrix[i, j] = p_value
            # print(label,feature,p_value)
        df_matrix.loc[:, label] = matrix[i, :]

    plt.figure()
    plt.title('feature hypothesis test p values')
    ax = sns.heatmap(matrix, xticklabels=2, yticklabels=False)
    ax.set_yticks([x + 0.5 for x in list(range(matrix.shape[0]))])
    ax.set_yticklabels(list(category_features), size=int(100 / matrix.shape[1]));
    ax.set_xticks([x + 0.5 for x in list(range(matrix.shape[1]))])
    ax.set_xticklabels(list(features), size=int(100 / matrix.shape[1]), rotation=90);
    plt.show()
    return df_matrix


def get_from_action_data(fname, chunk_size=100000,save='tail'):
    import pandas as pd
    import os
    reader = pd.read_csv(fname, iterator=True,low_memory=False)
    chunks = []
    loop = True
    while loop:
        try:
            print('run')
            chunk = reader.get_chunk(chunk_size).sort_values(by='datercv')
            if save=='tail':
                chunks.append(chunk.groupby('ID').tail(1))
            else:
                chunks.append(chunk.groupby('ID').head(1))
            print(chunk.shape)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    if save=='tail':
        df_ac = pd.concat(chunks, ignore_index=True).groupby('ID').tail(1)
    else:
        df_ac = pd.concat(chunks, ignore_index=True).groupby('ID').head(1)
    return df_ac


def get_raw_data(fname, feature, chunk_size=100000,IDs=[],columns=[]):
    import pandas as pd
    import os
    reader = pd.read_csv(fname, iterator=True,low_memory=False)
    chunks = []
    loop = True
    while loop:
        try:
            print('run')
            chunk = reader.get_chunk(chunk_size)
            if len(IDs)<1:
                if len(columns)<1:
                    select=chunk.loc[:,:].copy()
                else:
                    select=chunk.loc[:,columns].copy()

            else:
                if len(columns)<1:
                    select=chunk.loc[chunk[feature].isin(IDs),:].copy()
                else:
                    select=chunk.loc[chunk[feature].isin(IDs),columns].copy()

            chunks.append(select)
            print(select.shape)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac.reset_index(drop=True)


# kuikui
def forcast_plot(history, forcast, model='test'):
    axis_x = [x for x in range(start_x, end_x)]
    plt.figure(figsize=(15,6))
    plt.title(model)
    plt.plot(axis_x[:len(history)-overlap],history[:len(history)-overlap],'-o',color='blue',markerfacecolor='none', markeredgecolor='blue',
             markersize=4.0,linewidth=1.0,label='history')
    plt.plot(axis_x[len(history) - overlap:],forcast, '-o',color='red',markerfacecolor='none', markeredgecolor='red',
             markersize=4.0,linewidth=1.0, label='forcast')
    plt.legend(loc='upper left')
    plt.show()


def compute_dummy_feature(year,month,delta_month):
    year_month=str(int(year))+'-'+str(int(month))
    date=datetime.strptime(year_month, '%Y-%m').date()+ relativedelta(months=+delta_month)
    month=date.month
    quarter=(month-1)//3+1
    return date.year, date.month,quarter


def moving_new_window_preds(model, X_test_begin, test_length=12, dummy_feature=True):
    # dummy feature create
    flat_test_begin = X_test_begin.reshape(-1)
    if dummy_feature:
        seq = flat_test_begin[3]
        history_label = list(flat_test_begin[4:])

        for i in range(test_length):
            seq_new = seq + i
            year, month, quarter = compute_dummy_feature(2005, 1, seq_new)
            ready_test = np.array([year, month, quarter, seq_new] + history_label[i:])
            ready_test = np.reshape(ready_test, X_test_begin.shape)
            # print(ready_test.shape)
            prediction = model.predict(ready_test)
            prediction = np.reshape(prediction, (-1))
            # print(prediction.shape)
            history_label.append(prediction[0])

        return np.reshape(history_label[-1 * test_length:], (len(history_label[-test_length:]), 1))

    else:
        history_label = list(X_test_begin.reshape(-1, ))
        feature_len = len(history_label)
        for i in range(test_length):
            ready_test = np.array(history_label[i:]).reshape(X_test_begin.shape)
            prediction = model.predict(ready_test)
            prediction = np.array(prediction).reshape(-1)
            history_label.append(prediction[0])
        print(test_length)
        return history_label[-1 * test_length:]


def moving_new_window_preds_dummy(model, X_test_begin, y_test, test_length=12, new_prediction=12, dummy_feature=True):
    # dummy feature create
    flat_test_begin = X_test_begin[0]
    if dummy_feature:

        history_label = list(flat_test_begin[6:])
        seq = len(time_series) - test_length
        for i in range(test_length):
            seq_new = seq + i
            year, month, quarter = compute_dummy_feature(2005, 1, seq_new)
            ready_test = np.array([month, quarter] + list(X_test_begin[i][2:6]) + history_label[i:])
            ready_test = np.reshape(ready_test, (1, flat_test_begin.shape[0]))
            # print(ready_test.shape)
            prediction = model.predict(ready_test)
            prediction = np.reshape(prediction, (-1))
            # print(prediction.shape)
            history_label.append(prediction[0])

        return np.reshape(history_label[-1 * test_length:], (len(history_label[-test_length:]), 1))

    else:
        flat_test_begin = X_test_begin[-1, :]
        history_label = list(flat_test_begin[7:]) + [y_test[-1]]
        feature_len = len(history_label)
        seq = len(time_series)
        for i in range(new_prediction):
            seq_new = seq + i
            year, month, quarter = compute_dummy_feature(2005, 1, seq_new)
            ready_test = np.array([month, quarter] + list(X_test_begin[i][2:6]) + history_label[i:]).reshape(
                (1, flat_test_begin.shape[0]))
            prediction = model.predict(ready_test)
            prediction = np.array(prediction).reshape(-1)
            history_label.append(prediction[0])
        print(test_length)
        return history_label[-1 * new_prediction:]


def n_mean(y_error,n=3):
    quarter_array=[]
    q_sum=0
    for i in range(len(y_error)):
        q_sum+=y_error[i]
        if ((i+1)%3==0) :
            quarter_array.append(q_sum/3)
            q_sum=0
    return np.array(quarter_array)


def plot_error(y_error,title=' '):
    plt.figure()
    plt.title(title)
    plt.plot(y_error,'r-o')
    plt.ylabel('error for test_data')
    plt.grid()
    plt.savefig('pp_forecast/'+title+'.png')
    plt.show()
    quarter_error=n_mean(y_error,n=3)
    quarter_mae=   np.mean(np.abs(quarter_error))
    print('quarter error is',quarter_mae)
    return quarter_mae


def train_lgb(X_train, y_train, X_test, y_test):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    param1 = {}
    param1['metrics'] = 'rmse'
    gb_1 = lgb.train(param1, lgb_train, valid_sets=lgb_eval,
                     num_boost_round=4000, verbose_eval=300,
                     early_stopping_rounds=200)

    return gb_1


def conv1d_model(X_train,y_train,n_features=1,epochs=500):
    model = Sequential()
    model.add(Conv1D(filters=4, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], n_features)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=4, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], n_features)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    ES = [EarlyStopping(monitor='val_loss',patience=100,verbose=0,mode='auto')]

    model.fit(X_train, y_train, epochs=epochs, verbose=0,
             validation_split=0.1,
             callbacks=ES)
    return model


def prepare_data(data, lags=1, forecast=1,model='lstm'): # data is a np array
    """
    Create lagged data from an input time series
    lags: how many previous time steps as input X
    forecast: forecast at which time step ahead, i.e, how many time steps ahead in the future
    """
    X, y = [], []
    for row in range(len(data) - lags - forecast + 1):
        a = data[row:(row + lags), 0]
        X.append(a)
        y.append(data[row + lags + forecast -1, 0])
    X=np.array(X)
    if model =='lstm':
        return X.reshape((X.shape[0],1,X.shape[1])), np.array(y).reshape(-1)
    elif model =='1dconv':
        return X.reshape((X.shape[0],X.shape[1],1)), np.array(y).reshape(-1)
    else:
        return X.reshape((X.shape[0],X.shape[1])), np.array(y).reshape(-1)


def lstm_model(X_train,y_train,lags=12,epochs=1000):
    model = Sequential()
    model.add(LSTM(15,activation = 'relu', input_shape=(1, lags)))
    #model.add(Dense(20, input_dim=20, activation='relu'))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    ES = [EarlyStopping(monitor='val_loss',patience=100,verbose=0,mode='auto')]

    model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=0,
              validation_split=0.1,callbacks=ES)
    return model


def mlp_regression(X_train,y_train,X_test,y_test):
    seed = 10  # 9000
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(0.1))

    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Early stopping
    ES = [EarlyStopping(monitor='val_loss',patience=100,verbose=0,mode='auto')]

    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size = 16,
                        validation_data=(X_test, y_test),
                        #validation_split=0.1,
                        callbacks=ES,
                        verbose=0)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.grid()
    plt.show()
    return model


def moving_avg(array,half_window=1):
    array=np.array(array)
    new_array=array.copy()
    for i in range(half_window,len(array)-half_window):
        #new_array[i]=np.mean(array[i-half_window:i+half_window])
        new_array[i]=np.max(array[i-half_window:i+half_window])
    return new_array


def normalize_forecast(time_series,label_feature,_date_var_,state,coverage, epochs=1000, lags=24, test_length=24, rmse_ends=-12, model_type='lstm', \
                       validation=True, new_prediction=12):
    df_mth = time_series[[_date_var_, label_feature]]
    # set column YR-MTH as index
    df_mth.set_index(_date_var_, inplace=True)
    np.random.seed(1)

    # load the dataset
    data = df_mth.values
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data[:, -1:])

    # prepare the model data format

    forecast = 1
    if not validation:
        test_length = 3
    if model_type == 'lstm':
        X, y = prepare_data(dataset, lags, forecast, model=model_type)

        # split into train and test sets
        ntrain = len(X) - test_length  # 120
        X_train, X_test = X[0:ntrain, :], X[ntrain:, :]
        y_train, y_test = y[0:ntrain], y[ntrain:]
        model = lstm_model(X_train, y_train, lags=lags)
    elif model_type == '1dconv':
        X, y = prepare_data(dataset, lags, forecast, model=model_type)

        ntrain = len(X) - test_length  # 120
        X_train, X_test = X[0:ntrain, :], X[ntrain:, :]
        y_train, y_test = y[0:ntrain], y[ntrain:]
        y_test_true = scaler.inverse_transform([y_test])

        n_features = 1
        X_train_conv = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        X_test_conv = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        # define model
        model = conv1d_model(X_train_conv, y_train)

    # make predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])
    testPredict = scaler.inverse_transform(testPredict)
    y_test_true = scaler.inverse_transform([y_test]).reshape((-1))
    y_p = moving_new_window_preds(model, X_test[:1], test_length=test_length, dummy_feature=False)
    y_p = scaler.inverse_transform([y_p]).reshape((-1))
    y_p_test = y_p[:len(y_test_true)]
    if validation:
        save_folder = 'pp_forecast'
        algorithm = 'lstm'
        test_file = '%s-%s-%s_forecast_validation' % (state, coverage, model_type)
        #  test_based_on 2016 result
        testScore = math.sqrt(mean_squared_error(y_test_true[:rmse_ends], y_p_test[:rmse_ends]))
        print(model_type, 'test set rmse is', testScore)
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[lags:len(trainPredict) + lags, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (lags):len(dataset), :] = y_p_test.reshape((y_test_true.shape[0], 1))
        # plot baseline and predictions
        plt.figure(figsize=(15, 6))
        plt.title('rmse for test dataset of %s-%s-%s is %.2f' % (state, coverage, model_type, testScore))
        plt.plot(scaler.inverse_transform(dataset), 'g-o', label='true total loss')
        plt.plot(trainPredictPlot, 'b-o', label='train prediction')
        plt.plot(testPredictPlot, 'r-o', label='test prediction')
        plt.legend()
        plt.savefig(os.path.join(save_folder, test_file))
        plt.show()
        error_file = '%s-%s-%s_test_error' % (state, coverage, algorithm)

        plot_error(y_p_test[:rmse_ends] - y_test_true[:rmse_ends], title=error_file)

        return testScore
    else:
        total_length = test_length + new_prediction
        y_p = moving_new_window_preds(model, X_test[:1], test_length=total_length, dummy_feature=False)
        y_p = scaler.inverse_transform([y_p]).reshape((-1))
        forcast_plot(time_series[label_feature].values, y_p,
                     'Pure premium for %s-%s by %s' % (coverage, state, model_type))
        return y_p


def code_mean(data, cat_feature, real_feature):
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())


def dummy_regression_dataset_prepare(time_series, _date_var_, label_feature, lags=1, forecast=1, test_size=0):
    # --------dataset is ready to use for forecast model
    df_mth = time_series[[_date_var_, label_feature]]

    # ---------adding year feature which might be used later as index------
    df_mth['year'] = df_mth[_date_var_].dt.year

    # ---------adding month, quarter as features-----
    df_mth['month'] = df_mth[_date_var_].dt.month
    df_mth['quarter'] = df_mth[_date_var_].dt.quarter

    # ---------adding averaged pp over month and quarter level as features-------
    # ---------be careful, only the training data can be used-----
    test_index = len(df_mth) - test_size
    df_mth['month_average'] = list(map(code_mean(df_mth[:test_index], 'month', label_feature).get, df_mth.month))
    df_mth["quarter_average"] = list(map(code_mean(df_mth[:test_index], 'quarter', label_feature).get, df_mth.quarter))

    # ---------adding categorical feature is_first_quarter and is_third_quarter-------
    df_mth['is_first_quarter'] = (df_mth.quarter == 1) * 1
    df_mth['is_third_quarter'] = (df_mth.quarter == 3) * 1

    return df_mth


### Ray ##
def arules(data, supp, conf, lift, min_len, max_len, rhs):
    """
    A function for Association Rule Mining..

    Input:
        data - Data to mine
        supp - Support for association rule . Between 0.1 and 0.2 .
        conf - Confidence for association rule . Between 0.5 and 0.7 .
        min_len - Minimum length of mining rules
        max_len - Maximum length of mining rules
        rhs - The item to be in the riht hand side. Ex: Milk occures with chocolate, tea .
                Milk is in rhs and rest is in lhs. (chocolate, tea => Milk)

    Output: File containing the required fileds ( 'Confidence', 'Length', 'Lift', 'Support', 'lhs', 'rhs' )
    """
    import numpy as np
    import pandas as pd
    from apyori import apriori
    data = data.fillna("UNK")
    #########################################################
    dd = pd.DataFrame()
    for i in data.columns:
        dd[i] = str(i) + ' = ' + data[i].astype(str)
    #########################################################
    records = []
    col = data.shape[1]
    row = data.shape[0]
    for i in range(0, row):
        records.append([str(data.values[i, j]) for j in range(0, col)])
    association_rules = apriori(records,
                                min_support=supp,
                                min_confidence=conf,
                                min_lift=lift,
                                max_length=max_len,
                                min_length=min_len
                                )
    association_rules = list(association_rules)
    df = pd.DataFrame()
    for item in association_rules:
        pair = item[0]
        items = [x for x in pair]
        ############################################################################
        df = df.append({'Rules': '     ,      '.join(items), 'Support': str(item[1]),
                        'Confidence': str(item[2][0][2]), 'Lift': str(item[2][0][3]), 'Length': len(items)},
                       ignore_index=True)
        ############################################################################
    df.Confidence = df.Confidence.astype(np.float16)
    df.Lift = df.Lift.astype(np.float16)
    df.Support = df.Support.astype(np.float16)
    #########################################################
    dff = df[df['Rules'].str.contains(rhs)]
    def rhs_lhs(query, rhs):
        query = query
        stopwords = [rhs]
        querywords = query.split()
        resultwords = [word for word in querywords if word.lower() not in stopwords]
        result = '   '.join(resultwords)
        return (result)
    #########################################################
    dff['lhs'] = dff['Rules'].apply(lambda x: rhs_lhs(x, rhs))
    dff['rhs'] = rhs
    dff = dff.drop(['Rules'], axis=1).reset_index()
    #########################################################
    return dff


### Madhu ##
def readImage(path):
    import cv2
    """
    Read a image from a directory
    """
    image = cv2.imread(path)
    return image


def applyMeanFilter(image):
    import cv2
    """
    Mean filter over color image 
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
    figure_size = 9  # the dimension of the x and y axis of the kernal.
    new_image = cv2.blur(image, (figure_size, figure_size))
    return new_image


def applyMedianFilter(image):
    import cv2
    """
    Median filter over image 
    """
    figure_size = 9
    new_image = cv2.medianBlur(image, figure_size)
    return new_image


def applyLaplacianFilter(image):
    import cv2

    """
    Laplacian filter over image 

    """
    new_image = cv2.Laplacian(image, cv2.CV_64F)

    return new_image


def applyMaxRGBFilter(image):
    import cv2
    import numpy as np
    """
    MaxRGB filter over image 

    """
    (B, G, R) = cv2.split(image)
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
    new_image = cv2.merge([B, G, R])
    return new_image


def applyBilateralFilater(image):
    import cv2
    """
    Bilateral filter over image 

    """

    new_image = cv2.bilateralFilter(image, 9, 75, 75)
    return new_image


def binarize(image):
    """
    making image binarize

    """
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] < 50:
                image[i, j] = 255
            else:
                image[i, j] = 0
    return image


def makeGray(image):
    import cv2
    """
    making gray image 

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def dilation(image):
    import cv2
    import numpy as np
    """
    dilation operation over binary image 

    """
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(image, kernel, iterations=2)
    return img_dilation


def erosion(image):
    import cv2
    import numpy as np
    """
    erosion operation over binary image 

    """
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(image, kernel, iterations=1)
    return img_erosion


def computedCannyEdge(image, sigma=0.33):
    import cv2
    import numpy as np

    """
    computed canny edge over binary image 

    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def detect_corners(img, org_img):
    import cv2
    """
    corner detection over binary image 

    """
    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(img, 150, 0.01, 8)
    corners = np.int0(corners)
    return corners


def resizeImage(img):
    import cv2
    """
    dilation operation over binary image 

    """
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def displayImage(img):
    import cv2
    """
    display image 

    """
    res = resizeImage(img)
    cv2.imshow('Output', res)
    # Exiting the window if 'q' is pressed on the keyboard.
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return None


def binarizeImage(gray):
    """
    binarize image on the basis of some thrshold value

    """
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            if (gray[i, j] > 200):
                gray[i, j] = 255
            else:
                gray[i, j] = 0
    return gray


def filterSmallerObject(img, minArea):
    import cv2
    import numpy as np
    """
    filter smaller objects from image

    """
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = minArea
    # your answer image
    processed = np.zeros(img.shape)
    processed = processed.astype(np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            processed[output == i + 1] = 255
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # dilate = cv2.dilate(processed, kernel, iterations=3)
    return processed


def textExtract(img):
    import cv2
    import pytesseract
    """
    extract text from image 

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
    text = pytesseract.image_to_string(thresh, lang='eng',
                                       config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    return text


def CalDistance(p1, p2):
    import math
    """
    calculate distance between two points of an image 

    """
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return distance


def momentsHu(image):
    import cv2

    """
    calculate the HU's moments from an object segmented of an image

    """
    features = cv2.HuMoments(cv2.moments(image)).flatten()
    return features


def prewitt_horizontal_vertical(image):
    from skimage.filters import prewitt_h, prewitt_v

    """
    calculate the prewitt operator on horizontal and vertical direction of an image

    """
    # calculating horizontal edges using prewitt kernel
    edges_prewitt_horizontal = prewitt_h(image)
    # calculating vertical edges using prewitt kernel
    edges_prewitt_vertical = prewitt_v(image)
    return edges_prewitt_horizontal, edges_prewitt_vertical


def imrotate(image):
    import cv2
    """
    rotate image 

    """
    # Shape of image in terms of pixels.
    (rows, cols) = image.shape[:2]
    # getRotationMatrix2D creates a matrix needed for transformation.
    # We want matrix for rotation w.r.t center to 45 degree without scaling.
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    res = cv2.warpAffine(image, M, (cols, rows))
    return res


# ---------------------------------------------------------#

def affineTransform(image):
    import cv2
    import numpy as np
    """
    Affine Transform of an image

    """
    (rows, cols) = image.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    # warpAffine does appropriate shifting given the
    # translation matrix.
    res = cv2.warpAffine(image, M, (cols, rows))
    return res


def cannyEdge(image):
    import cv2
    """
    canny edge detection of an image

    """
    # Canny edge detection.
    edges = cv2.Canny(image, 100, 200)
    return edges


def imcrop(image, startRow, endRow, startCol, endCol):
    import cv2
    """
    crop an image based on requirement

    """
    croppedImage = image[startRow:endRow, startCol:endCol]

    return croppedImage


def addContrast(image):
    import cv2
    import numpy as np
    """
    Enhance the contrast of image

    """
    contrast_img = cv2.addWeighted(image, 2.5, np.zeros(image.shape, image.dtype), 0, 0)

    return contrast_img


def gaussian_blur(image):
    import cv2
    """
    blur image
    """
    blur_image = cv2.GaussianBlur(image, (7, 7), 0)

    return blur_image


def median_blur(image):
    import cv2
    """
    median blur image
    """
    blur_image = cv2.medianBlur(image, 5)

    return blur_image


def imgWrite(outputpath, filteredImage):
    import cv2
    """
    write image

    """
    cv2.imwrite(outputpath, filteredImage)


if __name__ == '__main__':
    il_all()
