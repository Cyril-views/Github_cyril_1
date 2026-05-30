"""
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
"""

"""
# Operations

There are lots of operations with pandas that will be really useful to you, but don't fall into any distinct category. Let's show them here in this lecture:
"""

import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()

"""
### Info on Unique Values
"""

df['col2'].unique()

df['col2'].nunique()

df['col2'].value_counts()

"""
### Selecting Data
"""

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]

newdf

"""
### Applying Functions
"""

def times2(x):
    return x*2

df['col1'].apply(times2)

df['col3'].apply(len)

df['col1'].sum()

"""
** Permanently Removing a Column**
"""

del df['col1']

df

"""
** Get column and index names: **
"""

df.columns

df.index

"""
** Sorting and Ordering a DataFrame:**
"""

df

df.sort_values(by='col2') #inplace=False by default

"""
** Find Null Values or Check for Null Values**
"""

df.isnull()

# Drop rows with NaN Values
df.dropna()

"""
** Filling in NaN values with something else: **
"""

import numpy as np

df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()

df.fillna('FILL')

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)

df

df.pivot_table(values='D',index=['A', 'B'],columns=['C'])

"""
# Great Job!
"""
