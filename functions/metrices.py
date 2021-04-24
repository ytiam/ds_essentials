from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
def auc_plot(prediction,labels,title_name=None):
    fpr, tpr, thresholds = roc_curve(np.array(labels),np.array(prediction))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(False)
    if title_name:
        plt.savefig(title_name)
    plt.show()
    return roc_auc




import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def important_score_plot(model,title='GB',plot_n=20):
    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
 
    
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(),model.feature_name())), columns=['Value','Feature'])

    plt.figure()
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(plot_n))
    plt.title('top %d important features'%plot_n)
    plt.tight_layout()
    plt.show()
    return feature_imp.sort_values('Value',ascending=False)
 




from sklearn.metrics import roc_curve, auc

def auc_compare(predictions,labels,index=''):
    if index=='':
        index=[str(i) for i in range(len(predictions))]
    predictions=list(predictions)

    roc_aucs=[0]*len(predictions)
    plt.figure()
    for i,prediction in enumerate(predictions):
        fpr, tpr, thresholds = roc_curve( labels[i],prediction)    
        roc_aucs[i] = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1, label=index[i] +' ROC curve (area = %0.2f)' % roc_aucs[i])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    roc_aucs_dict={}
    for i in range(len(index)):
        roc_aucs_dict[index[i]]=roc_aucs[i]
    return roc_aucs_dict


import itertools
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve
from sklearn.metrics import precision_recall_fscore_support

#The code for the below matrix is taken from sklearn documentation
#Defining the confusion matrix function
def plot_confusion_matrix(y_test,pred, classes=['Success','Failure'],
                          normalize=False,sticker_size=10,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,title_name=None):
    (precision, recall, f1_score,_)=precision_recall_fscore_support(y_test,pred,average='binary')
    cm = confusion_matrix(y_test,pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,size=sticker_size)
    plt.yticks(tick_marks, classes,size=sticker_size)
    plt.grid(False)



    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    acc=accuracy_score(y_test,pred)
    if title_name:
        plt.savefig(title_name)
    print('precision is %.3f'%precision)
    print('recall is %.3f'%recall)
    print('f1 score is %.3f'%f1_score)
    print('accuracy is %.3f'%acc)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
#from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
def f1(p,r):
    return 2*p*r/(p+r)
def precision_recall(y_test,pred,title_name=None):
    average_precision = average_precision_score(y_test, pred)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='red', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    precision, recall, thresholds = precision_recall_curve(y_test, pred)
    close_half = np.argmin(np.abs(thresholds-0.5))
    plt.plot(recall[close_half], precision[close_half], 'o', markersize=5,
    label="threshold 0.5", fillstyle="none", c='k', mew=2)

    f_array=np.array([2*precision[i]*recall[i]/max((precision[i]+recall[i]),0.0001) for i in range(len(recall))])
    max_f=np.argmax(f_array)
    plt.plot(recall[max_f], precision[max_f], '.', markersize=10,
    label="max f socre %.3f with threshold %.3f"%(f1(precision[max_f],recall[max_f]),thresholds[max_f]), 
             fillstyle="none", c='b', mew=2)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    #step_kwargs = ({'step': 'post'}
    #               if 'step' in signature(plt.fill_between).parameters
    #               else {})
    #plt.step(recall, precision, color='b', alpha=0.2,
    #         where='post')
    plt.plot(recall,precision)
    plt.grid(False)

    #plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    if title_name:
        plt.savefig(title_name)
    return  0
  

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
def best_threshold(y_test,pred,interval=0.02,pr_diff=0.05,according_index=2):
    prf=[]
    thres=[]
    best_score=0
    best_threshold=0.5
    for i in range(1,int(1//interval)):
        threshold=i*interval
        thres.append(threshold) 
        acc_=accuracy_score(y_test,pred>threshold)
        prf_=list(precision_recall_fscore_support(y_test,pred>threshold,average='binary'))
        prf_[3]=acc_
        prf_.append(threshold)
        if (abs(prf_[1]-prf_[0])< pr_diff) and (prf_[0]>0.1):
            if prf_[according_index]>best_score:
                    best_score=max(best_score,prf_[according_index])
                    best_threshold=threshold

    best_prf_=list(precision_recall_fscore_support(y_test,pred>best_threshold,average='binary'))
    best_prf_[3]=accuracy_score(y_test,pred>best_threshold)
    best_prf_.append(best_threshold)
    best_dict={}
    best_dict['precision']=best_prf_[0]
    best_dict['recall']=best_prf_[1]
    best_dict['f1']=best_prf_[2]
    best_dict['accuracy']=best_prf_[3]
    best_dict['threshold']=best_prf_[4]

    
    return best_dict
