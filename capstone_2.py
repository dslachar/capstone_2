import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt



'create list of years'
def create_years_list(start_year, end_year):
    year = start_year
    year_list = []
    while year <= end_year:
        year_list.append(year)
        year += 1
    return year_list

years = create_years_list(1999, 2018)

'load data frame from each year'
df_1999 = pd.read_csv('data/transactions_1999.csv')
df_2000 = pd.read_csv('data/transactions_2000.csv')
df_2001 = pd.read_csv('data/transactions_2001.csv')
df_2002 = pd.read_csv('data/transactions_2002.csv')
df_2003 = pd.read_csv('data/transactions_2003.csv')
df_2004 = pd.read_csv('data/transactions_2004.csv')
df_2005 = pd.read_csv('data/transactions_2005.csv')
df_2006 = pd.read_csv('data/transactions_2006.csv')
df_2007 = pd.read_csv('data/transactions_2007.csv')
df_2008 = pd.read_csv('data/transactions_2008.csv')
df_2009 = pd.read_csv('data/transactions_2009.csv')
df_2010 = pd.read_csv('data/transactions_2010.csv')
df_2011 = pd.read_csv('data/transactions_2011.csv')
df_2012 = pd.read_csv('data/transactions_2012.csv')
df_2013 = pd.read_csv('data/transactions_2013.csv')
df_2014 = pd.read_csv('data/transactions_2014.csv')
df_2015 = pd.read_csv('data/transactions_2015.csv')
df_2016 = pd.read_csv('data/transactions_2016.csv')
df_2017 = pd.read_csv('data/transactions_2017.csv')
df_2018 = pd.read_csv('data/transactions_2018.csv')

'crate list of data frames'
dfs = [df_1999, df_2000, df_2001, df_2002, df_2003, df_2004, df_2005, df_2006,
       df_2007, df_2008, df_2009, df_2010, df_2011, df_2012, df_2013, df_2014,
       df_2015, df_2016, df_2017, df_2018]

'create a main dataframe'
main_df = pd.concat(dfs)

'create a sector dataframe'
sector_df = main_df[['fiscal_year','activity_description',
                    'dac_category_name', 'dac_sector_name']]

'remove other and administrative costs categories from dataframe'
sector_df = sector_df[sector_df.dac_category_name != 'Other']
sector_df = sector_df[sector_df.dac_category_name != 'Administrative Costs']

'remove null values'
sector_df = sector_df.dropna()


'create a numpy array'
sector_arr = sector_df.values

'show counts by category and sector'
category_counts = sector_df.dac_category_name.value_counts().reset_index().rename(columns={'index': 'Category', 0: 'Count'})
sector_counts = sector_df.dac_sector_name.value_counts().reset_index().rename(columns={'index': 'Sector', 0: 'Count'})

'''countplot by category'''
# fig,ax = plt.subplots(1,1)
# ax = sns.countplot(y='dac_category_name', data=sector_df, order = sector_df['dac_category_name'].value_counts().index)
# fig.set_size_inches(8,8)
# #plt.show()

x = sector_df['activity_description']
y = sector_df['dac_category_name']

'train/test split'
x_train, x_test, y_train, y_test = train_test_split(x,y)


'''Naive Bayes Pipeline'''
NB_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('NB_clf', MultinomialNB())
])

'''SVM Pipeline'''
SVM_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('SVM_clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None))
])

'''Test Data Transform Pipeline'''
test_transform = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer())

])



'''transform test data'''
docs_new = x_test
'''Tokenizing test data'''
count_vect = CountVectorizer()
x_new_counts = count_vect.fit_transform(docs_new)

'''TfidfTransformer test data'''
tfidf_transformer = TfidfTransformer()
x_new_tfidf = tfidf_transformer.fit_transform(x_new_counts)






NB_model = NB_clf.fit(x_train, y_train)
#NB_test_transform = test_transform(x_test)
# 'predictions'
NB_predicted = NB_model.predict(x_test)

print('Naive Bayes Accuracy Score: {}'.format(np.mean(NB_predicted == y_test)))

SGD_model = SGD_clf.fit(x_train,y_train)

SGD_predicted = SGD_model.predict(x_test)

print('SGD Accuracy Score: {}'.format(np.mean(SGD_predicted == y_test)))



# '''SVM'''
# svm_clf = SVC()
# svm_clf.fit(x_train_tfidf, y_train)
#
# '''transform test data'''
# # docs_new = x_test
# # x_new_counts = count_vect.transform(docs_new)
# # x_new_tfidf = tfidf_transformer.transform(x_new_counts)
#
# '''predictions'''
# svm_predicted = svm_clf.predict(x_new_tfidf)
#
# print('SVM Accuracy Score: {}'.format(np.mean(svm_predicted == y_test)))
#
# '''
# Naive Bayes Accuracy Score: 0.8524514338575393
# SVM Accuracy Score: 0.30583543765782434
# '''






































