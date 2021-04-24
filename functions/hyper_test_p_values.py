from sklearn import preprocessing
from sklearn.feature_selection import chi2,f_classif
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def hyper_test_pvalue(label,feature,dropna=True):
    df=pd.DataFrame()
    df['label']=label.apply(lambda x: 'label'+str(x))
    df['feature']=feature.values
    if dropna:
        no_nan=df.dropna().copy()
        #print('nan rate for %s is %.2f'%('feature',1-no_nan.shape[0]/df.shape[0]))
    else:
        #print("replace NAN with string 'NAN'")
        df.loc[:,'feature']=df.loc[:,'feature'].replace(np.nan,'NAN',regex=True)
        df.loc[:,'label']=df.loc[:,'label'].replace(np.nan,'NAN',regex=True)
        no_nan=df.copy()
    if  len(no_nan)<20:
        return np.nan
    if 'int' not in str(no_nan['label'].dtype):
        le_label=preprocessing.LabelEncoder()
        temp_list=list(map(lambda x: str(x),no_nan['label'].unique()))
        le_label.fit(temp_list)      
        no_nan.loc[:,'label']=le_label.transform(no_nan['label'])
    if str(feature.dtype) in ('category','objective','bool') or feature.nunique()<0.05*len(feature):
        le = preprocessing.LabelEncoder()
        no_nan['feature']=list(map(lambda x: str(x),no_nan['feature']))
        le.fit(no_nan['feature'].unique())
        no_nan.loc[:,'feature']=le.transform(no_nan.loc[:,'feature'])
        
        (score,p_value)=chi2(no_nan['feature'].values.reshape(-1,1),no_nan['label'])
    else:
        (score,p_value)=f_classif(no_nan['feature'].values.reshape(-1,1),no_nan.loc[:,'label'])
    return p_value[0]

def heat_map_p_value(raw_df,category_features,features):
    df_matrix=pd.DataFrame(index=features)
    matrix=-1*np.ones([len(category_features),len(features)])
    for i,label in enumerate(category_features):
        for j,feature in enumerate(features):        
            p_value=hyper_test_pvalue(raw_df[label],raw_df[feature])
            matrix[i,j]=p_value
            #print(label,feature,p_value)
        df_matrix.loc[:,label]=matrix[i,:]

    plt.figure()
    plt.title('feature hypothesis test p values')
    ax = sns.heatmap(matrix, xticklabels=2, yticklabels=False)
    ax.set_yticks([x + 0.5 for x in list(range(matrix.shape[0]))])
    ax.set_yticklabels(list(category_features), size = int(100 / matrix.shape[1]));
    ax.set_xticks([x + 0.5 for x in list(range(matrix.shape[1]))])
    ax.set_xticklabels(list(features), size = int(100 / matrix.shape[1]),rotation=90);
    plt.show()
    return df_matrix