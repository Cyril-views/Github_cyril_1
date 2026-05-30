"""
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Pandas Data Visualization Exercise

This is just a quick exercise for you to review the various plots we showed earlier. Use **df3** to replicate the following plots. 
"""

import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')
%matplotlib inline

df3.info()

df3.head()

"""
** Recreate this scatter plot of b vs a. Note the color and size of the points. Also note the figure size. See if you can figure out how to stretch it in a similar fashion. Remeber back to your matplotlib lecture...**
"""

"""
** Create a histogram of the 'a' column.**
"""

"""
** These plots are okay, but they don't look very polished. Use style sheets to set the style to 'ggplot' and redo the histogram from above. Also figure out how to add more bins to it.***
"""

"""
** Create a boxplot comparing the a and b columns.**
"""

"""
** Create a kde plot of the 'd' column **
"""

"""
** Figure out how to increase the linewidth and make the linestyle dashed. (Note: You would usually not dash a kde plot line)**
"""

"""
** Create an area plot of all the columns for just the rows up to 30. (hint: use .ix).**
"""

"""
## Bonus Challenge!
Note, you may find this really hard, reference the solutions if you can't figure it out!
** Notice how the legend in our previous figure overlapped some of actual diagram. Can you figure out how to display the legend outside of the plot as shown below?**

** Try searching Google for a good stackoverflow link on this topic. If you can't find it on your own - [use this one for a hint.](http://stackoverflow.com/questions/23556153/how-to-put-legend-outside-the-plot-with-pandas)**
"""

"""
# Great Job!
"""
