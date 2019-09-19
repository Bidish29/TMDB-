#!/usr/bin/env python
# coding: utf-8

# 
# # TMDB Investigation.
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# 
# ## Introduction
# 
# I have selected the TMDB data for my analysis. I will be exploring the data based on the movie genres. 
# 
# Some quesions that I would like to ask are,
# 
#     1. Which genre is the most profitable and which is the least?
#     2. How many movies in each genre got a rating of more than 7?
#     3. How has the profit been affected over the years?
# 

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# # 1.Data Wrangling
# 
# 

# ## 1.1 Investigating the data
# 

# In[2]:


# Reading csv into Pandas Dataframe and store in dataset variable
dframe = pd.read_csv('tmdb-movies.csv')


# In[3]:


dframe.describe()


# In[4]:


#Finding out the number of rows and columns
dframe.shape


# In[5]:


#Printing out information about the data
dframe.info()


# After printing out the information about the dataset, we see that values are missing for, homepage,director,tagline,keywords,production_companies,overview,cast,imdb_id,genre. 
# For our investigation, we do not need the imdb_id,cast,homepage,director,tagline,overview,release_date(as we already have a column with release year),production_company,budget_adj,revenue_adj,keywords columns
# For the remaining columns I will need to find a way tp update the values.

# In[6]:


#Finding the datatypes of each column
dframe.dtypes


# In[7]:


dframe.columns


# In[8]:


dframe.describe()


# # 2. Data Cleaning

# ## 2.1 Removing the unwanted columns

# For our investigation, we do not need the imdb_id,cast,homepage,director,tagline,overview,release_date(as we already have a column with release year),production_company,budget_adj,revenue_adj,keywords columns. So I will be dropping those columns now.

# In[9]:


#Dropping the unwanted columns
dframe=dframe.drop(['imdb_id','cast', 'homepage', 'director', 'tagline', 'keywords', 'overview','production_companies', 'release_date','budget_adj',
       'revenue_adj'], axis=1)
dframe.head()


# ## 2.2 Handling the missing values

# We see that, there are missing values in the budget,revenue and runtime columns. I will be replacing these columns with the mean.

# In[10]:


# mean value of budget
mean_budget = dframe['budget'].mean(skipna=True)
mean_budget


# In[11]:


# mean value of revenue
mean_revenue = dframe['revenue'].mean(skipna=True)
mean_revenue


# In[12]:


# mean value of runtime
mean_runtime = dframe['runtime'].mean(skipna=True)
mean_runtime


# In[13]:


# Replacing the budget nulls with the mean

dframe['budget']=dframe.budget.mask(dframe.budget == 0,mean_budget)
dframe.describe()


# In[14]:


# Replacing the revenue nulls with the mean

dframe['revenue']=dframe.revenue.mask(dframe.revenue == 0,mean_revenue)
dframe.describe()


# In[15]:


# Replacing the runtime nulls with the mean

dframe['runtime']=dframe.runtime.mask(dframe.runtime == 0,mean_runtime)
dframe.describe()


# In[16]:


#Analysing the new dataset
dframe.shape


# In[17]:


##Printing out information about the data
dframe.info()


# In[18]:


#Printing a few columns
dframe.head()


# In[19]:


#Splitting the genres

s = dframe['genres'].str.split('|').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'genres'
del dframe['genres']
df = dframe.join(s)


# In[20]:


#Printing a few columns to check if the split occurred
df.head()


# In[21]:


#Information about the new dataset
df.info()


# In[22]:


#Checking the number of new rows and columns
df.shape


# ## 2.3 Checking for NULL values

# In[23]:


#Checking if there are any null values
df.isnull().sum()


# In[24]:


#Dropping the null values
df=df.dropna(axis=0,how='any')


# In[25]:


#Checking to see if there are any more null values
df.isnull().sum()


# ##### In this new dataset, there are duplicates as I am exploring it based on the Genres and when I had split it based on the Genres values were duplicated

# <a id='eda'></a>
# # 3. Exploratory Data Analysis
# 
# 
# 

# Let us now visualise the data we have with plots

# #### 1. Let's see how the votes are distributed

# In[26]:


df['vote_average'].plot(kind='hist',title='Distribution of votes',legend=True,figsize=(10,10));


# We see that movies with an average vote of 5 or more are more.

# #### 2. Now let's check the popularity of the genres

# In[27]:


df.groupby('genres')['popularity'].mean().plot.bar(legend=True,figsize=(10,10));


# We see that on an average the most popular genre is Adventure and the least is Documentary

# #### 3. Let's check the runtimes

# In[28]:


df.groupby('genres')['runtime'].mean().plot(kind='barh',title='Distribution of the runtime',legend=True,figsize=(10,10));


# We see that history has the highest runtime

# ## Question 1: Which genre is the most profitable and which is the least?

# In[29]:


df.groupby('genres')['popularity'].mean()


# In[30]:


#Let's add a new column called profit to the dataframe. Let's calculate it based on revenue and budget.

df['profit']=df['revenue']-df['budget']


# In[31]:


#let's check if the new column is added or not.
df.head()


# We see that a new column called profit is added at the end.

# In[32]:


#Now, let us group the profit by genre
df.groupby('genres')['profit'].mean()


# We see that the profits are not easily readable. Let's convert the profit in terms of millions.

# In[33]:


#We divide by a million
df['profit']=df['profit']/1000000


# In[34]:


#Let's check the dataframe now
df.head()


# #### I will be further dividing the dataset and using only those columns as needed to answer the questions as it will make it faster and easier to run and analyse

# In[35]:


# I need  the profit and genre columns to answer this question.
df1=df[['original_title', 'profit','genres']]
df1.head()


# Now it is much easier to view and analyse.  Now, lets visualize and find the answers to the quesion.

# In[36]:


#Finding out the mean and grouping by genre
df2 = df1.groupby(['genres']).mean()
df2


# In[37]:


#Sorting the values
df2.sort_values('profit', ascending=False, inplace = True )


# In[38]:


#Plotting the graph
df2[['profit']].plot.barh(stacked=True, title = 'Genres by profit (US$ million)', figsize=(10, 8));


# From the plot, we see that the Adventure is most profitable and the least profitable genre is Documentary.

# ## Question 2: How many movies in each genre got a rating of more than 7?

# In[39]:


#Finding out the ratings above 7
df3 = df[df['vote_average']>=7]   


# In[40]:


df3.describe()


# In[41]:


# I need the genre and vote_average columns
df4=df3[['vote_average','genres','original_title']]
df4.head()


# In[42]:


#Sorting the values
df5 = (pd.DataFrame(df4.groupby('genres').original_title.nunique())).sort_values('original_title', ascending=False )
df5


# In[43]:


df5.describe()


# In[44]:


# Plotting a pie chart
df5['original_title'].plot(kind='pie',figsize=(15,15),title = 'Movies with rating over 7');


# From the pie cart, we see that Drama has 775 movies rated above 7 where as TV movie has the least with 21.

# ## Question 3: How has the profit been affected over the years?

# In[45]:


# I need the profit and the release_year
df7=df[['release_year','profit','original_title']]
df7.head()


# In[46]:


#Grouping by year and taking the average
df8=df7.groupby(['release_year']).mean()


# In[47]:


df8


# In[48]:


#Plotting the graph
plt.xlabel('Year',fontsize=10)
plt.ylabel('Profit (US$ million) ',fontsize=10)

plt.bar(df8.index,df8['profit']);
plt.title('Profits over decades (US$ million)')


# We see that the profit has increased over the years.

# <a id='conclusions'></a>
# ## Conclusions
# 
# ### 6.1 Limitations of dataset
# The are a number of limitations with the TMDB, which are caused by:
# 
#     -missing data
#     -restrictd to this is only a sample dataset
#     -missing other feature that would be interesting for analysis
# 
# There is a lot of data missing for the budget,revenue and runtime for which I had to adjust the dataset by taking the mean of each column and appending the mean to the missing rows. Although this helped improve the data, it was not a correct representation of the actual budget,revenue or the runtime of the movies.
# 
# The number of movies released each year are far greater than the scope of this dataset. This data could be random sample or it could be biased to a specific genre or could be unbiased.
# 
# We see that there is a vote count and vote average, if we were to have other columns that would say if the voters were the critics or the general public, it would have been easier to see if the movie was popular with the masses or if it was a critically accalimed movie.
# 
# ### 6.2 Answering the questions
# I had asked the following questions at the beginning.
# 
#     1. Which genre is the most profitable and which is the least?
#     2. How many movies in each genre got a rating of more than 7?
#     3. How has the profit been affected over the years?
# 
#     Ans 1: From the plot, we see that the Adventure is most profitable and the least profitable genre is Documentary.
# 
#     Ans 2: From the pie cart, we see that Drama has 775 movies rated above 7. This shows that people love watching more of Drama.Also along with Drama,Comedy,Action also have a good number of movies with ratings higher than 7 which indicates that the audience has received the movie well, but its not the same in case of genres like documentary,foreign or TV movie. 
# 
#     Ans 3: From the plot, we see that in the 1960s, the mean profit peaked at 30-40 million US dollars. This has significantly imroved in the 2010s where the mean profit is almost 60 million US dollars.
