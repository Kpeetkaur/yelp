#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


import pandas as pd


# In[5]:


import pandas as pd
data =pd.read_csv('/Users/kamalpreetkaur/Downloads/yelp data set.csv')


# In[ ]:





# In[6]:


data.dtypes


# In[7]:


data.shape
data.values


# In[8]:


data.head(20)


# In[9]:


data[data.duplicated()]


# In[10]:


data= data.drop_duplicates()


# In[11]:


data.isnull().values.any()


# In[12]:


data.head(10)


# In[13]:


data['length'] = data['text'].apply(len)
data.head()


# In[14]:


data['length'].plot(bins=100, kind='hist')


# In[15]:


data.length.describe()


# In[16]:


data[data['length']==1]['text'].iloc[0]


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


data[data['length']==755]['text'].iloc[0]


# In[19]:


sns.countplot(y ='stars', data=data)


# In[20]:


g = sns.FacetGrid(data=data, col='stars', col_wrap=3)


# In[21]:


g = sns.FacetGrid(data=data, col='stars', col_wrap=5)
g.map(plt.hist, 'length', bins =20, color ='m')


# In[22]:


data_1 =data[data['stars']==1]
data_5 =data[data['stars']==5]


# In[23]:


data_1


# In[24]:


data_3= data[data['stars']==3]
data_3


# In[25]:


data_1_5 =pd.concat([data_1,data_5])
data_1_5


# In[26]:


data_1_5.info()


# In[27]:


print( '1-stars percentage =', (len(data_1)) / len(data_1_5)*100, "%")


# In[28]:


print( '5-stars percentage=', (len(data_5)) / len(data_1_5)*100, "%")


# In[29]:


sns.countplot(data_1_5['stars'], label = "count")


# In[30]:


import string
string.punctuation


# In[31]:


Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'


# In[32]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[33]:


Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[34]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[35]:


Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[38]:


Test_punc_removed_join


# In[39]:


Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[40]:


Test_punc_removed_join_clean


# In[41]:


mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'


# In[42]:


challege = [ char     for char in mini_challenge  if char not in string.punctuation    ]
challenge = ''.join(challege)
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[44]:


print(vectorizer.get_feature_names())


# In[45]:


print(X.toarray())  


# In[46]:


mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())


# In[47]:


def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[ ]:


data_clean = data_1_5['text'].apply(message_cleaning)


# In[50]:


print(data_clean[0])


# In[51]:


print(data_1_5['text'][0])


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(data_1_5['text'])


# In[53]:


print(vectorizer.get_feature_names())


# In[54]:


yelp_countvectorizer.shape


# In[55]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = data_1_5['stars'].values


# In[56]:


label


# In[57]:


NB_classifier.fit(yelp_countvectorizer, label)


# In[58]:


testing_sample = ['amazing food! highly recommmended']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[59]:


testing_sample = ['shit food, made me sick']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[60]:


X = yelp_countvectorizer
y = label


# In[61]:


X.shape


# In[62]:


y.shape


# In[ ]:





# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[64]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[65]:


from sklearn.metrics import classification_report, confusion_matrix


# In[66]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[67]:


y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[68]:


print(classification_report(y_test, y_predict_test))


# In[69]:


yelp_countvectorizer


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:





# In[ ]:




