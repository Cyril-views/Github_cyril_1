"""
___

<a href='http://www.pieriandata.com'> <img src='../../Pierian_Data_Logo.png' /></a>
___
"""

"""
# SF Salaries Exercise - Solutions

Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.
"""

"""
** Import pandas as pd.**
"""

import pandas as pd

"""
** Read Salaries.csv as a dataframe called sal.**
"""

sal = pd.read_csv('Salaries.csv')

"""
** Check the head of the DataFrame. **
"""

sal.head()

"""
** Use the .info() method to find out how many entries there are.**
"""

sal.info() # 148654 Entries

"""
**What is the average BasePay ?**
"""

sal['BasePay'].mean()

"""
** What is the highest amount of OvertimePay in the dataset ? **
"""

sal['OvertimePay'].max()

"""
** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **
"""

sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']

"""
** How much does JOSEPH DRISCOLL make (including benefits)? **
"""

sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']

"""
** What is the name of highest paid person (including benefits)?**
"""

sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].max()] #['EmployeeName']
# or
# sal.loc[sal['TotalPayBenefits'].idxmax()]

"""
** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**
"""

sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].min()] #['EmployeeName']
# or
# sal.loc[sal['TotalPayBenefits'].idxmax()]['EmployeeName']

## ITS NEGATIVE!! VERY STRANGE

"""
** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **
"""

sal.groupby('Year').mean()['BasePay']

"""
** How many unique job titles are there? **
"""

sal['JobTitle'].nunique()

"""
** What are the top 5 most common jobs? **
"""

sal['JobTitle'].value_counts().head(5)

"""
** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **
"""

sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1) # pretty tricky way to do this...

"""
** How many people have the word Chief in their job title? (This is pretty tricky) **
"""

def chief_string(title):
    if 'chief' in title.lower():
        return True
    else:
        return False

sum(sal['JobTitle'].apply(lambda x: chief_string(x)))

"""
** Bonus: Is there a correlation between length of the Job Title string and Salary? **
"""

sal['title_len'] = sal['JobTitle'].apply(len)

sal[['title_len','TotalPayBenefits']].corr() # No correlation.

"""
# Great Job!
"""
