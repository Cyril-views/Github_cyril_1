"""
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
"""

"""
# DataFrames

DataFrames are the workhorse of pandas and are directly inspired by the R programming language. We can think of a DataFrame as a bunch of Series objects put together to share the same index. Let's use pandas to explore this topic!
"""

import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())

df

"""
## Selection and Indexing

Let's learn the various methods to grab data from a DataFrame
"""

df['W']

# Pass a list of column names
df[['W','Z']]

# SQL Syntax (NOT RECOMMENDED!)
df.W

"""
DataFrame Columns are just Series
"""

type(df['W'])

"""
**Creating a new column:**
"""

df['new'] = df['W'] + df['Y']

df

"""
** Removing Columns**
"""

df.drop('new',axis=1)

# Not inplace unless specified!
df

df.drop('new',axis=1,inplace=True)

df

"""
Can also drop rows this way:
"""

df.drop('E',axis=0)

"""
** Selecting Rows**
"""

df.loc['A']

"""
Or select based off of position instead of label 
"""

df.iloc[2]

"""
** Selecting subset of rows and columns **
"""

df.loc['B','Y']

df.loc[['A','B'],['W','Y']]

"""
### Conditional Selection

An important feature of pandas is conditional selection using bracket notation, very similar to numpy:
"""

df

df>0

df[df>0]

df[df['W']>0]

df[df['W']>0]['Y']

df[df['W']>0][['Y','X']]

"""
For two conditions you can use | and & with parenthesis:
"""

df[(df['W']>0) & (df['Y'] > 1)]

"""
## More Index Details

Let's discuss some more features of indexing, including resetting the index or setting it something else. We'll also talk about index hierarchy!
"""

df

# Reset to default 0,1...n index
df.reset_index()

newind = 'CA NY WY OR CO'.split()

df['States'] = newind

df

df.set_index('States')

df

df.set_index('States',inplace=True)

df

"""
## Multi-Index and Index Hierarchy

Let us go over how to work with Multi-Index, first we'll create a quick example of what a Multi-Indexed DataFrame would look like:
"""

# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

hier_index

df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df

"""
Now let's show how to index this! For index hierarchy we use df.loc[], if this was on the columns axis, you would just use normal bracket notation df[]. Calling one level of the index returns the sub-dataframe:
"""

df.loc['G1']

df.loc['G1'].loc[1]

df.index.names

df.index.names = ['Group','Num']

df

df.xs('G1')

df.xs(['G1',1])

df.xs(1,level='Num')

"""
# Great Job!
"""
