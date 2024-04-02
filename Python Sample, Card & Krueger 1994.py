# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:34:14 2023

@author: Keegan Sweet

Description:
    
This is a Python recreation of a popular labor market study conducted by economists, David Card and Alan B. Krueger, published in the American Economic Review journal in 1994. The study, and Krueger's prior work, challenge conventional labor economic theory: that unemployment will rise with minimum wage. 

The study observes a New Jersey minimum wage increase effective April, 1992. Nearby Eastern Pennsylvania did not experience a rise in minimum wage, and was selected as a control group for its high level of homogeneity with New Jersey. The study found that employment did not drop as a result of minimum wage, and had even marginally risen. The unconventional nature of the results lead to the article being cited 4,826 times since 1994 (Google Scholar citation metrics). The study is also a popular exhibit of difference-in-difference analysis, one clear time-shift, and homogenous populations with low confounding variables (fast-food restaurants).

This sample demonstrates basic Python capability within Spyder, the Pandas library, multiple regression models, and difference-in-differences analysis. Three models are created with increasing exogenous variables to demonstrate various statistical effects. Run the models separately to view the respective results in the terminal (model1, model2, model3).

Last Updated: Fri, Aug 25 11:55EST, 2023'
"""

#importing numpy & pandas libraries as aliases
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

#Assigning data file to dataframe in pandas
dataset = pd.read_csv('njmin3.csv')

#It's my first time operating Python. From previous software experience, I believe I understand the syntax,at least in the context of Spyder and included libraries. It "talks" something like this: variableassign=object.function(call/execute, specify_option).subfunction()

#Viewing data, setting view options:
dataset.describe()
pd.set_option('display.max_columns', None)
dataset.describe()

#Showing which columns contain instances of null:
dataset.isnull().any()

#Importing a simple imputer from sci-kit learn:

#Telling the imputer to target instances of numpy NA's and convert them to mean. Then, assigning the impution to a missingvalues variable:
missingvalues = SimpleImputer(missing_values=np.nan, strategy='mean')

#Overwriting the missingvalues variable with an instance of itself that specifies which columns in the dataframe it should apply to:
missingvalues = missingvalues.fit(dataset[['fte', 'demp']])

#Permitting missingvalues variable to transform dataframe:
dataset[['fte', 'demp']] = missingvalues.transform(dataset[['fte', 'demp']])

#Checking result:
dataset.isnull().any()

#Setting independent/dependent X & Y by locating values from the table index:
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values

#Importing a statistical modeling library:

#Using statsmodels to calculate the constants and add as a virtual column. Note that the virtual column is only applied to X and not the dataframe: X(:,3)=/=Y(:,3)
X = sm.add_constant(X)

#1st OLS multiple regression model:
model1 = sm.OLS(Y, X).fit()

#Naming the X's and Y's:
model1.summary(yname="FTE", xname=("intercept", "New Jersey",
               "After April 92", "NJ after April 92"))

#Introduce more variables to test 2nd model. Let's introduce the restaurant classes:
X = dataset.loc[:, ['NJ', 'POST_APRIL92', 'NJ_POST_APRIL92',
                    'bk', 'kfc', 'roys', 'wendys']].values
Y = dataset.loc[:, 'fte'].values
X = sm.add_constant(X)
model2 = sm.OLS(Y, X).fit()
model2.summary(yname=("FTE"), xname=("intercept", "New Jersey",
               "After April 92", "NJ after April 92", "BK", "KFC", "Roys", "Wendys"))

#Dummy-class variables are causing strong multicollinearity in our second model. Let's make an alteration to the independent variables to clean it up:
X = dataset.loc[:, ['NJ', 'POST_APRIL92',
                    'NJ_POST_APRIL92', 'bk', 'kfc', 'wendys']].values
X = sm.add_constant(X)
model2 = sm.OLS(Y, X).fit()
model2.summary(yname=("FTE"), xname=("intercept", "New Jersey",
               "After April 92", "NJ after April 92", "BK", "KFC", "Wendys"))

#Removed Roys. 2nd model now contains acceptable level of multicollinearity
#Introducing more independent variables has kept the FTE correlation with the wage increase constant while increasing the significance level.

#Let's add a few more x-variables for our 3rd model. Now we'll include whether a restaurant was franchised or company owned and if it was located in South or central NJ:
X = dataset.loc[:, ['NJ', 'POST_APRIL92', 'NJ_POST_APRIL92',
                    'bk', 'kfc', 'wendys', 'co_owned', 'centralj', 'southj']].values
X = sm.add_constant(X)
model3 = sm.OLS(Y, X).fit()
model3.summary(yname=("FTE"), xname=("intercept", "New Jersey", "After April 92",
               "NJ after April 92", "BK", "KFC", "Wendys", 'co_owned', 'centralj', 'southj'))

#More x-variables have resulted in a marginally higher significance level while again maintaining the regression coefficient of the effect of wage on FTE. However, at 95% confidence, the results are not significant enough to attribute the marginal increase in employment to the result of the wage increase. The one certainty this case-study yields, is that an increase in minimum wage DID NOT result in lower employment as stated by traditional economic theory
