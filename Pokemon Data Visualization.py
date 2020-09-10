#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[13]:


df = pd.read_csv('D:\\UDEMY\\Fundamental_Data_Analysis_Viz\\Pokemon.csv', index_col=0, encoding='unicode-escape')
df.head()


# In[14]:


sns.lmplot(x = 'Attack', y = 'Defense', data=df)
plt.show()


# In[16]:


sns.lmplot(x = 'Attack', y= 'Defense', data=df, fit_reg=False, hue='Stage')


# Boxplot

# In[17]:


df_copy =df.drop(['Total', 'Stage', 'Legendary'], axis=1)
sns.boxplot(data=df_copy)


# Violinplot

# In[19]:


sns.violinplot(data=df_copy)
plt.show()


# Primary Type VS Attack 

# In[25]:


plt.figure(figsize=(10,8))

sns.violinplot(x='Type 1', y='Attack', data=df)
plt.show()


# Heatmap
# 

# In[26]:


corr = df_copy.corr()


# In[27]:


sns.heatmap(corr)


# In[28]:


sns.distplot(df.Attack, color='blue')


# Pokemon count based on primary type

# In[30]:


sns.countplot(x='Type 1', data=df)
plt.xticks(rotation=-45)


# Compare attack and defense values 

# In[31]:


sns.jointplot(df.Attack, df.Defense, kind='kde', color='lightblue')


# In[4]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} plotly')


# In[7]:


import plotly.express as px
import pandas as pd
import numpy as np


# In[6]:


df = pd.read_csv('D:\\UDEMY\\Fundamental_Data_Analysis_Viz\\Pokemon.csv', index_col=0, encoding='unicode-escape')
df.head()


# In[10]:


#Scatter plot to compare attack and defense scores of each pokemmon

fig = px.scatter(df, x='Attack', y='Defense')
fig.show()


# In[12]:


# add pokemon name in hover data
fig = px.scatter(df, x='Attack', y='Defense', color ='Stage', hover_data=['Name'])
fig.show()


# In[14]:


fig = px.scatter(df, x='Attack', y='Defense', color='Type 1', size='Total', hover_data=['Name'])
fig.show()


# In[20]:


x = list(range(df.shape[0]))
y = df['Total'].sort_values()

fig = px.line(df,x,y,labels={'x':'Pokemon Rank', 'y':'Total Score'},hover_data=['Name'])
fig.show()


# In[24]:


df_copy = df.drop(['Name','Type 1', 'Type 2', 'Total', 'Stage', 'Legendary'], axis=1)
fig = px.box(df_copy)
fig.show()


# In[25]:


fig = px.violin(df_copy)
fig.show()


# In[27]:


#violin plot to compare distribution of attacks to the Pokemon's primary type

fig = px.violin(df, x='Type 1', y='Attack', color='Type 1')
fig.show()


# In[30]:


#heatmap to view corrolation between attributes

corr = df_copy.corr()

labels = list(df_copy)

fig = px.imshow(corr, x=labels, y=labels)

fig.show()


# In[31]:


#histogram to see distribution of attack values for pokemon in a particular range

fig = px.histogram(df, x='Attack')
fig.show()


# In[32]:


#density heat map for attack and defense scores

fig = px.density_heatmap(df, x='Attack', y='Defense')
fig.show()


# In[ ]:




