"""
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
"""

"""
# Missing Data

Let's show a few convenient methods to deal with Missing Data in pandas:
"""

import numpy as np
import pandas as pd

df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})

df

df.dropna()

df.dropna(axis=1)

df.dropna(thresh=2)

df.fillna(value='FILL VALUE')

df['A'].fillna(value=df['A'].mean())

"""
# Great Job!
"""
