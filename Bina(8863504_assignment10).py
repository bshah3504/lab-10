#!/usr/bin/env python
# coding: utf-8

# In[30]:


# 1.Task 1: Dataset Selection 
# a.Choose two datasets from the provided repositories. 
# Answer: I have chosen the following two datasets from the provided repositories:

# Dataset 1: Titanic Dataset
# Dataset 2: College Scoreboard Data


# b.Justify your selection for each dataset based on its relevance to machine learning tasks. Include a brief paragraph explaining the dataset's potential for analysis and its suitability for machine learning applications.
# Answer: Dataset 1: Titanic Dataset

# Justification: The Titanic dataset comprises data on passengers, including information on their age, 
# sex, ticket type, and whether or not they survived. This dataset serves as a standard for binary classification 
# problems and is a well-known example in the fields of data analysis and machine learning. The possibility for analysis 
# rests in examining trends and elements that contributed to the passengers' survival during the sad tragedy. Based on 
# the provided features, we can use a variety of machine learning algorithms, such as logistic regression, decision trees, 
# or random forests, to forecast passenger survivability. The dataset is excellent for machine learning applications due to its
# historical context and clearly stated target variable.

# Justification: The College Scorecard offers data by field of study as well as data at the institution level. 
#                In-depth details concerning these data are provided in the technical data publications. 
#                The data dictionary has distinct worksheets with institution-level, field-of-study, and group maps outlining the timing characteristics of each data element.


# In[ ]:


# 2.Task 2: Data Exploration with Python 
# a.Perform exploratory data analysis (EDA) using Python for the first dataset. 
# b.Generate summary statistics, identify data types, and visualize the data distribution to gain insights into the dataset.


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the dataset into a DataFrame
df = pd.read_csv('C:/Users/HOME/Desktop/titanic.csv',encoding='ISO-8859-1')


# In[4]:


print("Preview of the dataset:")
print(df.head())


# In[5]:


print("\nDataset Information")
print(df.info())


# In[6]:


# Summary statistics
print("\nSummary Statistics")
print(df.describe())


# In[7]:


# Visualize data distribution
print("\nHistograms:")
df.hist(figsize=(10,8))
plt.title("Histogram")
plt.tight_layout()
plt.show()


# In[ ]:


# 3.Task 3: Data Preprocessing with Python 
# a.Preprocess the data from the first dataset using Python. 
# b.Handle missing values, outliers, and perform feature engineering when necessary to prepare the data for machine learning models.


# In[8]:


sns.boxplot(x='Age', data=df)
plt.xlabel('Survived')
plt.ylabel('Age')
plt.title('Age Distribution based on Survival')
plt.show()


# In[9]:


# Handling missing values
print("\nMissing values:")
print(df.isnull().sum())


# In[10]:


# Handling outliers
df = df[df['Age'] < 60]  # Remove outliers where age is greater than 60


# In[11]:


# Feature engineering
df['age_squared'] = df['Age'] ** 2  # Add new feature: age_squared


# In[12]:


df.drop(['Cabin','Name','Ticket','PassengerId'], axis =1, inplace = True) 
print(df.head())


# In[13]:


df['Sex']=pd.factorize(df['Sex'])[0]
df['Embarked']=pd.factorize(df['Embarked'])[0]


# In[ ]:


# 4.Task 4: Implement Machine Learning Models with Python 
# a.Implement at least two different machine learning models (e.g., SVM, Random Forest, Neural Network) for the first dataset using Python. 
# b.Evaluate and compare the performance of each model using appropriate metrics to determine the most suitable model for the dataset.


# In[14]:


X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[19]:


svm_model = SVC(kernel='linear', random_state=40)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
SVMAccuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVM: {SVMAccuracy:.2f}')
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)


# In[ ]:


# 5.Task 5: Visualization with Python 
# a.Create meaningful visualizations (e.g., scatter plots, heatmaps, bar charts) for the first dataset using Python. 
# b.Use libraries like Matplotlib, Seaborn, or Plotly to create clear and insightful visual representations of the dataset.


# In[20]:


random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)
RandomAccuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Random Forest: {RandomAccuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)


# In[23]:


print("\nCorrelation Heatmap")
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.show()


# In[24]:


# Visualization : Survival Count based on Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Count by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# In[26]:


# Visualization: Box plot of Age based on Survived
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# In[27]:


# Visualization: Bar chart of Survival Count
plt.figure(figsize=(8, 6))
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival Count by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# In[29]:


# Visualization : Scatter plot of Survival Count
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Scatter Plot of Age vs. Fare by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# In[ ]:




