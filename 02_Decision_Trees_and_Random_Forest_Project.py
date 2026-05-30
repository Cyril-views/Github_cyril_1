"""
<a href="https://colab.research.google.com/github/Cyril-views/Github_cyril_1/blob/ML-Projects/02_Decision_Trees_and_Random_Forest_Project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Random Forest Project

For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.

Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.

We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.

Here are what the columns represent:
* credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* installment: The monthly installments owed by the borrower if the loan is funded.
* log.annual.inc: The natural log of the self-reported annual income of the borrower.
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* fico: The FICO credit score of the borrower.
* days.with.cr.line: The number of days the borrower has had a credit line.
* revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
* inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
* delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
"""

"""
# Import Libraries

**Import the usual libraries for pandas and plotting. You can import sklearn later on.**
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd

# Direct link to the raw clean dataset
url = 'https://raw.githubusercontent.com/yash-kh/Random-Forest-Project-Lending-Club-Dataset/master/loan_data.csv'

# Read the data
loans = pd.read_csv(url)

# Verify the load
loans.info()

"""
## Get the Data

** Use pandas to read loan_data.csv as a dataframe called loans.**
"""

loans.head()

"""
** Check out the info(), head(), and describe() methods on loans.**
"""

loans.info()

loans.describe()

"""
# Exploratory Data Analysis

Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.

** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**

*Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*
"""

plt.hist(loans['not.fully.paid'], bins = 100)

"""
** Create a similar figure, except this time select by the not.fully.paid column.**
"""

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30,alpha=0.5,color='blue',label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30,alpha=0.5,color='red',label='not.fully.paid=0')
plt.legend()

plt.xlabel('FICO')

"""
** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **
"""

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')

"""
** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**
"""

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

"""
** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**
"""

sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')

"""
# Setting up the Data

Let's get ready to set up our data for our Random Forest Classification Model!

**Check loans.info() again.**
"""

loans.info()

"""
## Categorical Features

Notice that the **purpose** column as categorical

That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.

Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.

**Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**
"""

cat_feats = ['purpose']

"""
**Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**
"""

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

loans.columns

"""
## Train Test Split

Now its time to split our data into a training set and a testing set!

** Use sklearn to split your data into a training set and a testing set as we've done in the past.**
"""

from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis = 1)
Y = final_data['not.fully.paid']
x_train, x_test, y_train, y_test =  train_test_split(X,Y, test_size =0.3)

x_train.head()

"""
## Training a Decision Tree Model

Let's start by training a single decision tree first!

** Import DecisionTreeClassifier**
"""

from sklearn.tree import DecisionTreeClassifier

"""
**Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**
"""

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

"""
## Predictions and Evaluation of Decision Tree
**Create predictions from the test set and create a classification report and a confusion matrix.**
"""

from sklearn.metrics import classification_report,confusion_matrix

predictions = dtc.predict(x_test)

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

"""
## Training the Random Forest model

Now its time to train our model!

**Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**
"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(x_train, y_train)

rfc

"""
## Predictions and Evaluation

Let's predict off the y_test values and evaluate our model.

** Predict the class of not.fully.paid for the X_test data.**
"""

rfc_predictions = rfc.predict(x_test)

"""
**Now create a classification report from the results. Do you get anything strange or some sort of warning?**
"""

print(classification_report(y_test, rfc_predictions))

print(confusion_matrix(y_test, rfc_pred))

"""
**Show the Confusion Matrix for the predictions.**
"""

print(confusion_matrix(y_test, rfc_predictions))

"""
**What performed better the random forest or the decision tree?**
"""

print("### Decision Tree Classifier Classification Report:\n" + str(classification_report(y_test, predictions)))
print("### Decision Tree Classifier Confusion Matrix:\n" + str(confusion_matrix(y_test, predictions)))
print("\n### Random Forest Classifier Classification Report:\n" + str(classification_report(y_test, rfc_predictions)))
print("### Random Forest Classifier Confusion Matrix:\n" + str(confusion_matrix(y_test, rfc_predictions)))

print("\n### Comparison:\n")
print("**Overall Accuracy:** The Random Forest Classifier shows a higher overall accuracy (0.84) compared to the Decision Tree (0.74).")
print("\n**Performance on Class 0 (Fully Paid Loans):** Both models perform very well in predicting loans that **will be fully paid**. The Random Forest is slightly better with a precision of 0.84 and recall of 1.00, meaning it correctly identified almost all fully paid loans, with very few false positives.")
print("\n**Performance on Class 1 (Not Fully Paid Loans):** This is where the difference is significant, and it's crucial given the imbalance in the dataset (many more fully paid loans than not fully paid).")
print("\n*   **Decision Tree:** It has a precision of 0.22 and a recall of 0.24 for the 'not.fully.paid' class. This means out of all loans it predicted as 'not fully paid', only 22% actually were. And it only managed to identify 24% of the actual 'not fully paid' loans (110 out of 462).")
print("*   **Random Forest:** While its precision for 'not.fully.paid' is higher (0.33), its recall is extremely low (0.01). This means it only identified a tiny fraction (1%) of the actual 'not fully paid' loans (5 out of 443). It is very conservative in predicting a loan as 'not fully paid'.")
print("\n**Conclusion:**\n")
print("The **Random Forest Classifier** has a much higher overall accuracy because it is very good at predicting the majority class (loans that are fully paid). However, it is almost completely unable to identify the minority class (loans that are not fully paid), resulting in a very low recall for that class. It basically predicts almost every loan will be paid back.")
print("\nThe **Decision Tree Classifier**, while having lower overall accuracy, performs slightly better at identifying the 'not.fully.paid' loans (higher recall for class 1 compared to Random Forest). It catches more of the risky loans, even if its overall precision and accuracy are lower. ")
print("\n**Which performed better depends on your objective:**\n")
print("*   If your primary goal is to maximize overall accuracy and you don't mind missing most of the defaulted loans, the **Random Forest** is technically 'better'.")
print("*   If your primary goal is to identify risky loans (minimize false negatives for 'not fully paid'), the **Decision Tree** is slightly better, but still not great. ")
print("\nGiven the imbalance in the target variable, a more robust model would likely involve techniques like oversampling the minority class, undersampling the majority class, or using algorithms designed for imbalanced classification to improve the recall for 'not.fully.paid' without sacrificing too much precision.")

"""
# Great Job!
"""
