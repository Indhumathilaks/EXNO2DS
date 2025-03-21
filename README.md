# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = '/content/drive/MyDrive/Colab Notebooks/titanic_dataset.csv' 
dt = pd.read_csv(file_path)
dt
```
![image](https://github.com/user-attachments/assets/c3eea8ec-5ff1-4180-a600-d5a66353067f)
```
dt.info()
```
![image](https://github.com/user-attachments/assets/84f28767-dbc5-43e7-9a88-c842304d6778)
```
num_rows, num_columns = dt.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
```
![image](https://github.com/user-attachments/assets/b26800e8-dac6-4a22-8f23-5d559d89836a)
```
dt.set_index('PassengerId', inplace=True)
dt.head()
```
![image](https://github.com/user-attachments/assets/345c396d-b14b-4644-8b3f-89be94c304c9)
```
dt.describe()
```
![image](https://github.com/user-attachments/assets/9026e26b-a901-45cd-b18f-21e5dbcbf2b5)
```
# Count the frequency of each unique value in the 'Survived' column
survived_counts = dt['Survived'].value_counts()
survived_percentages = dt['Survived'].value_counts(normalize=True) * 100
print("Survived Counts:")
print(survived_counts)
print("\nSurvived Percentages:")
print(survived_percentages)
```
![image](https://github.com/user-attachments/assets/7fc11c8f-a736-4299-94a7-59a47dba6123)
```
# Count the frequency of each unique value in the 'Sex' column
sex_counts = dt['Sex'].value_counts()
sex_percentages = dt['Sex'].value_counts(normalize=True) * 100
print("Sex Counts:")
print(sex_counts)
print("\nSex Percentages:")
print(sex_percentages)
```
![image](https://github.com/user-attachments/assets/57ed496f-5ad1-4a91-9cce-6da78430a4d9)
```
# Count the frequency of each unique value in the 'Pclass' column
pclass_counts = dt['Pclass'].value_counts()
pclass_percentages = dt['Pclass'].value_counts(normalize=True) * 100
print("Pclass Counts:")
print(pclass_counts)
print("\nPclass Percentages:")
print(pclass_percentages)
```
![image](https://github.com/user-attachments/assets/baf9d2e0-a3a5-444c-8a54-a2f00b1c5127)
```
# Count the frequency of each unique value in the 'Embarked' column
embarked_counts = dt['Embarked'].value_counts()
embarked_percentages = dt['Embarked'].value_counts(normalize=True) * 100
print("Embarked Counts:")
print(embarked_counts)
print("\nEmbarked Percentages:")
print(embarked_percentages)
```
![image](https://github.com/user-attachments/assets/c123ecb0-6c97-4808-a1a1-a721a42400ed)
```
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style for seaborn
sns.set(style="whitegrid")
# Plot the count of survivors
sns.countplot(x='Survived', data=dt)
plt.title('Survival Distribution')
plt.show()
```
![image](https://github.com/user-attachments/assets/ef7d5fc8-e069-4271-9110-0f4946e85421)
```
# Plot the count of passengers by class
sns.countplot(x='Pclass', data=dt)
plt.title('Passenger Class Distribution')
plt.show()
```
![image](https://github.com/user-attachments/assets/929cb57d-24b4-4cec-81aa-446d61ea0631)
```
import seaborn as sns
import matplotlib.pyplot as plt
# Set the style for seaborn
sns.set(style="whitegrid")
# Create a count plot for the 'Survived' column
sns.countplot(x='Survived', data=dt)
plt.title('Survival Distribution (Univariate Analysis)')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/user-attachments/assets/5c3762bd-093b-445a-af48-43005c2d39e6)
```
# Identify unique values in the 'Pclass' column
unique_pclass = dt['Pclass'].unique()
print("Unique values in 'Pclass':", unique_pclass)
```
![image](https://github.com/user-attachments/assets/7a8f2458-1498-4b6c-99b5-1470e9b18c9b)
```
# Rename the 'Sex' column to 'Gender'
dt.rename(columns={'Sex': 'Gender'}, inplace=True)
# Verify the column name change
print(dt.columns)
```
![image](https://github.com/user-attachments/assets/1473a500-0588-426c-aaa0-794b23eaafcb)
```
import seaborn as sns
import matplotlib.pyplot as plt
# Use catplot to analyze the relationship between 'Survived' and 'Pclass'
sns.catplot(x='Pclass', hue='Survived', data=dt, kind='count', height=5, aspect=1.5)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/user-attachments/assets/13482322-294f-46f2-b390-2ab22181f4a0)
```
# Create a count plot for 'Pclass' with 'Survived' as the hue
fig, ax1 = plt.subplots(figsize=(8, 5))
graph = sns.countplot(x='Pclass', hue='Survived', data=dt, ax=ax1)
# Add labels to the bars
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2, height + 20.8, height, ha="center")
# Set plot title and labels
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right')
plt.show()
```
![image](https://github.com/user-attachments/assets/25fff4f1-b3b4-45f3-8543-b68ce4f154a3)
```
# Create a boxplot for 'Age' vs 'Survived'
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=dt)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
```
![image](https://github.com/user-attachments/assets/ab2bd3d7-6a40-47ac-9409-30164441cfe8)
```
import seaborn as sns
import matplotlib.pyplot as plt
# Create a boxplot for 'Age' vs 'Pclass' with 'Gender' as the hue
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', hue='Gender', data=dt)
plt.title('Age Distribution by Passenger Class and Gender')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Age')
plt.legend(title='Gender')
plt.show()
```
![image](https://github.com/user-attachments/assets/b99fa462-9e76-448c-8ee6-99c8bc38aace)
```
# Use catplot to analyze 'Pclass', 'Survived', and 'Gender'
sns.catplot(x='Pclass', hue='Survived', col='Gender', data=dt, kind='count', height=5, aspect=1)
plt.suptitle('Survival Count by Passenger Class and Gender', y=1.02)
plt.show()
```
![image](https://github.com/user-attachments/assets/be73f425-725c-4ebb-ab42-8c3d1444d225)
```
# Calculate the correlation matrix for numerical columns only
corr = dt.select_dtypes(include=np.number).corr()
# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/94be06a4-625a-4097-b3e6-0003bbe58fea)
```
# Create a pairplot
sns.pairplot(dt, hue='Survived', height=2.5)
plt.suptitle('Pairplot of Numerical Columns Colored by Survival Status', y=1.02)
plt.show()
```
![image](https://github.com/user-attachments/assets/66050c0e-de87-48aa-bb6e-6adaa33a6a5c)


# RESULT
Thus, the Exploratory Data Analysis on the given data set was performed successfully.
