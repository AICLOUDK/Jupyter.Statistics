<center>
    <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width="300" alt="cognitiveclass.ai logo">
</center>


#### Import the required libraries we need for the lab.



```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
```

#### Read the dataset in the csv file from the URL



```python
boston_df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv')
```

#### Add your code below following the instructions given in the course to complete the peer graded assignment



```python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr
import statsmodels.api as sm

# Preview dataset
print("First five rows of the dataset:")
print(housing_data.head())

print("\nDataset info:")
print(housing_data.info())

print("\nSummary statistics:")
print(housing_data.describe())

# 1.Median value of owner-occupied homes (MEDV)
plt.figure(figsize=(8,6))
sns.boxplot(y=housing_data['MEDV'])
plt.title('Boxplot of Median Home Values (MEDV)')
plt.ylabel('Median value in $1000\'s')
plt.show()

# 2. Bar plot for the Charles River variable (CHAS)
plt.figure(figsize=(6,4))
sns.countplot(x='CHAS', data=housing_data)
plt.title('Count of Tracts Bounding Charles River')
plt.xlabel('Bounds Charles River (1=True, 0=False)')
plt.ylabel('Count')
plt.show()

# 3. Discretize AGE variable into three groups
def age_group(age):
    if age <= 35:
        return '35 or younger'
    elif age <= 70:
        return '36-70'
    else:
        return '70 or older'

housing_data['AGE_GROUP'] = housing_data['AGE'].apply(age_group)

# Boxplot of MEDV vs AGE_GROUP
plt.figure(figsize=(8,6))
sns.boxplot(x='AGE_GROUP', y='MEDV', data=housing_data)
plt.title('Median Home Value vs Age Groups')
plt.xlabel('Owner-Occupied Units Built Before 1940')
plt.ylabel('Median value in $1000\'s')
plt.show()

# 4. Scatter plot: NOX vs INDUS
plt.figure(figsize=(8,6))
sns.scatterplot(x='INDUS', y='NOX', data=housing_data)
plt.title('Nitric Oxides Concentration vs Non-Retail Business Acres')
plt.xlabel('Proportion of Non-Retail Business Acres (INDUS)')
plt.ylabel('Nitric Oxides Concentration (NOX)')
plt.show()

# 5. Histogram of PTRATIO
plt.figure(figsize=(8,6))
sns.histplot(housing_data['PTRATIO'], bins=20, kde=True)
plt.title('Distribution of Pupil-Teacher Ratio')
plt.xlabel('Pupil-Teacher Ratio (PTRATIO)')
plt.ylabel('Frequency')
plt.show()

# -Statistical Tests 
# 1. Difference in MEDVbounding
medv_chas_1 = housing_data[housing_data['CHAS'] == 1]['MEDV']
medv_chas_0 = housing_data[housing_data['CHAS'] == 0]['MEDV']

t_stat, p_value = ttest_ind(medv_chas_1, medv_chas_0, equal_var=False)
print("T-test for MEDV by CHAS (Charles River boundary):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Significant difference in median home values between the two groups.")
else:
    print("Result: No significant difference in median home values between the two groups.")
print("\n")

# 2. Difference in MEDV across AGE groups (ANOVA)
group1 = housing_data[housing_data['AGE_GROUP'] == '35 or younger']['MEDV']
group2 = housing_data[housing_data['AGE_GROUP'] == '36-70']['MEDV']
group3 = housing_data[housing_data['AGE_GROUP'] == '70 or older']['MEDV']

F_stat, p_value = f_oneway(group1, group2, group3)
print("ANOVA for MEDV across Age Groups:")
print(f"F-statistic: {F_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Significant differences exist among the age groups.")
else:
    print("Result: No significant difference among the age groups.")
print("\n")

# 3. Relationship between NOX and INDUS (correlation)
corr_coef, p_value = pearsonr(housing_data['NOX'], housing_data['INDUS'])
print("Pearson correlation between NOX and INDUS:")
print(f"Correlation coefficient: {corr_coef:.4f}")
print(f"P-value: {p_value:.4f}")
if abs(corr_coef) < 0.3:
    print("Conclusion: Very weak or no linear relationship.")
elif abs(corr_coef) < 0.7:
    print("Conclusion: Moderate linear relationship.")
else:
    print("Conclusion: Strong linear relationship.")
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
print("\n")

# 4. Impact of DIS (distance to Boston employment centers) on MEDV (Regression)
X = housing_data[['DIS']]
X = sm.add_constant(X)  # adding a constant for intercept
Y = housing_data['MEDV']

model = sm.OLS(Y, X).fit()
print("Regression analysis: Impact of Distance to Boston Employment Centers on Median Home Value")
print(model.summary())

# analyze conclude;
# at the coefficient 'DIS' as well as p-value.
# If p < 0.05, DIS have effect on MEDV.
# Then, the coefficient indicates the direction.



```
