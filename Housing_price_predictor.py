#!/usr/bin/env python
# coding: utf-8

# ## Real_Estate Price Prediction

# In[1]:


import pandas as pd


# In[2]:


housing_df = pd.read_csv("data.csv")
housing_df.head()


# In[3]:


housing_df.info()


# In[4]:


housing_df['CHAS'].value_counts()


# In[5]:


housing_df.describe()


# ## Train-Test Splitting

# In[6]:


import numpy as np


# In[7]:


# spliting dataset into train and test set by directly using inbuild function for sklearn
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state =42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[8]:


# stratifing the test and train data set so that they contain equal proportions of all the different values
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits =1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df['CHAS']):
    strat_train_set = housing_df.loc[train_index]
    strat_test_set = housing_df.loc[test_index]


# In[9]:


strat_test_set.describe()


# In[10]:


housing_df = strat_train_set.copy() # housing_df is made equal to the train set


# ## Looking for Correlations

# In[11]:


corr_matrix = housing_df.corr()


# In[12]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[13]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'PTRATIO', 'LSTAT']
scatter_matrix(housing_df[attributes], figsize=(12, 8))


# ## Trying out Attribute Combinations

# In[14]:


housing_df['TAXRM'] = housing_df['TAX']/housing_df['RM'] # Tax per room


# In[15]:


housing_df.info()


# In[16]:


housing_df.plot(kind='scatter', x='TAXRM', y='MEDV')


# In[17]:


housing_df = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing attributes

# To take casre of missing attributes, we can do one of these:
#     1. Get rid of the missing data points
#     2. Get rid of the entire attribute
#     3. Set the missing value to some value(0, mean or median)

# In[18]:


from sklearn.impute import SimpleImputer # Imputation
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_df) # fit housing_df into imputer


# In[19]:


x = imputer.transform(housing_df) # transform housing_df into imputated dataset


# In[28]:


housing_tr = pd.DataFrame(x, columns = housing_df.columns) # tranformed dataframe housing_tr with imputation


# In[21]:


housing_tr.describe()


# ## Feature Scaling

# ## Creating a Pipeline

# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy = "median")), ('std_scaler', StandardScaler()),])


# In[29]:


housing_num_tr = my_pipeline.fit_transform(housing_df)
housing_num_tr # fully tranformed dataset fitted into pipeline


# ## Selecting the model

# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[56]:


some_data = housing_df.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[57]:


list(some_labels)


# ## Evaluating the Model

# In[58]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[59]:


rmse


# ## Cross Validation for better Evaluation

# In[62]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[63]:


rmse_scores


# In[64]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[65]:


print_scores(rmse_scores)


# ## Saving the Model

# In[66]:


from joblib import dump, load
dump(model, 'Prediction_Model.joblib') 

