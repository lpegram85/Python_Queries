import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve


#Binomial logistic regression

df = pd.read_csv(r'C:\Users\Lisa.Pegram\Downloads\data_Athlete2.csv', header=0)
#dropping the na fields
df=df.dropna()
df.describe()
#print(df.shape)
#print(list(df.columns))

#checking to make sure I don't have to group time
print(df['NOC'].unique())
print(df.corr())
plt.pcolor(df.corr(method='pearson'))
plt.show()

#df.corr()['NOC'].plt()

df['Win'].value_counts()
sns.countplot(x='Win',data=df,palette='hls')
plt.show()
#plt.savefig('Count Plot')

#checking out the means across the different variables
print(df.groupby('Win'))

pd.crosstab(df['Sport'],df['Win']).plot(kind='bar')
plt.title('Win Frequency by Sport')
plt.xlabel('Win Bit')
plt.ylabel('Frequency of Sport')
plt.savefig('Frequency_of_Sport')
plt.show()



table = pd.crosstab(df['Sport'],df['Win'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of City vs Win')
plt.xlabel('Win Bit')
plt.ylabel('Proportion of Wins')
plt.show()

#correlation between variables



#CREATING DUMMY VARIABLES FOR CATEGORICAL VARIABLES
cat_vars=['Sex','NOC', 'Sport']

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    data1=df.join(cat_list)
    df=data1

#print(cat_list)

data_vars=df.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]

#print(to_keep)
data_final=df[to_keep]


data_final_vars=data_final.columns.values.tolist()
y='Win'
X=[i for i in data_final_vars if i not in y]
#X=data_final[columns]

print(X)
print(data_final[X])

logreg = LogisticRegression()
rfe = RFE(logreg,n_features_to_select=8,verbose=1)
rfe = rfe.fit(data_final[X], data_final[y].values.ravel())
print(rfe.support_)
print(rfe.ranking_)

colnames = data_final[X].columns





# Define dictionary to store our rankings
ranks = []
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
ranks= ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
#
#
pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\Python'
filenameOut = 'to_keep.csv'
pathOut = pathnameOut + "/" + filenameOut


fileOut = open(pathOut, 'w')
fileOut.write(str(to_keep))

filenameOut2 = 'ranks.csv'
pathOut2 = pathnameOut + "/" + filenameOut2

fileOut = open(pathOut, 'w')
fileOut.write(str(ranks))

fileOut = open(pathOut2, 'w')
fileOut.write(str(ranks))


cols=[
    'NOC_ANG',
    'NOC_EGY',
    'NOC_GUA',
    'NOC_HKG',
    'NOC_ISR',
    'NOC_ISV',
    'NOC_PUR',
    'NOC_SEN',
    'NOC_BAR',
    'NOC_CYP',
    'NOC_LUX',
    'NOC_LIB',
    'NOC_GHA',
    'NOC_URS',
    'Sport_Baseball',
    'NOC_GAB',
    'NOC_GDR',
    'NOC_EUN',
    'NOC_USA',
    'Sport_Curling',
    'Sport_Handball',
    'Sport_Taekwondo'

]

X=data_final[cols]

sns.heatmap(X.corr())

y=data_final['Win']



logit_model=sm.Logit(y,X)
result=logit_model.fit(method='bfgs',maxiter=200)
print(result.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

selector = RFECV(estimator=logreg,scoring='neg_mean_squared_error')
selector.fit(X, y)
print('Optimal number of features: %d' % selector.n_features_)



y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))



#need to create a confusion matrix, tells yoy how many you got right or wrong we know what false positivty means
#when something is the same in actual and predict then it is a true positice
#when it is a false positive it is when the assign a positive when it isn't
#true negative is when it is wrong and it is actually wrong the others are opposite


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#

print(classification_report(y_test, y_pred))



logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#with(sn, cor(post.count, gold.spent))