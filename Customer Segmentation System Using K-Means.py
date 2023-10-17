#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation System Using K-Means

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[16]:


df = pd.read_csv('Customers.csv')


# In[17]:


df.head(10)


# In[18]:


df.shape


# In[19]:


df.info()


# In[20]:


X = df.iloc[:, [3, 4]].values


# In[21]:


X


# ### Perform Elbow Method to find Optimal No. of Clusters

# In[22]:


from sklearn.cluster import KMeans
wcss = []


# In[23]:


for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[24]:


plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()


# ### Training a model using Unsupervised Learning Algorithm

# In[25]:


kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)


# In[26]:


y_kmeans = kmeansmodel.fit_predict(X)


# In[27]:


plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s= 80, c = "red", label='Customer 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s= 80, c = "blue", label='Customer 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s= 80, c = "yellow", label='Customer 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s= 80, c = "cyan", label='Customer 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s= 80, c = "black", label='Customer 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s= 100, c= 'magenta', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# You can find this project on <a href='https://github.com/Vyas-Rishabh/Customer_Segmentation_System_Using_K-Means_InternSavy'><b>Github.</b></a>
