# Employee Churn Analysis

This project analyzes employee churn using historical data to identify contributing factors and predict future churn. The goal is to provide actionable insights for talent retention.

## Project Goal

The primary objective is to analyze and predict employee churn using historical HR data, enabling proactive strategies to improve employee retention.

## Dataset

The project utilizes historical HR data from the `employeedata.tbl_hr_data` table in Google BigQuery. This data includes employee demographics, roles, tenure, salary, and churn status.

## Methodology

The project follows a structured data analysis pipeline:

### 1. Data Retrieval Using BigQuery
The initial step involves connecting to Google BigQuery and retrieving the necessary dataset.
```python
# Connect to BigQuery
from google.cloud import bigquery
from google.colab import auth

# Authenticate
auth.authenticate_user()

# Initialize the client for BigQuery
project_id = 'churn-cloud-project-431011'
client = bigquery.Client(project=project_id, location='US')

# Get the dataset
dataset_ref = client.dataset('employeedata', project='churn-cloud-project-431011')
dataset = client.get_dataset(dataset_ref)
table_ref = dataset.table('tbl_hr_data')
table = client.get_table(table_ref)
```
The code above establishes a connection to the BigQuery client and references the specific table containing the HR data.

### 2. Data Exploration
The dataset's schema is examined to understand its structure, data types, and key columns relevant to churn analysis, such as employee tenure, department, and exit reasons.
```python
# Display the schema of the HR data table
table.schema
```
This command outputs the schema, providing an overview of available data fields.

### 3. Data Cleaning and Preprocessing
Data cleaning is performed to ensure the accuracy of the analysis. This involves handling missing values and removing duplicates. The BigQuery table is converted to a Pandas DataFrame for easier manipulation.
```python
# Example of handling missing values
import pandas as pd

# Convert BigQuery table to a Pandas DataFrame
query = """
SELECT * FROM `churn-cloud-project-431011.employeedata.tbl_hr_data`
"""
df = client.query(query).to_dataframe()

# Drop rows with missing values in critical columns
df = df.dropna(subset=['Age', 'Tenure', 'Department'])

# Display cleaned data
df.head()
```
The snippet above loads the data into a DataFrame and removes rows with missing data in 'Age', 'Tenure', or 'Department' columns.

### 4. Churn Analysis and Feature Engineering
Patterns in the data are analyzed, and new features are engineered to better capture churn indicators. For example, employee tenure is converted from months to years, and a binary feature for high salary is created.
```python
# Convert Tenure from months to years
df['Tenure_Years'] = df['Tenure'] / 12

# Feature engineering for churn prediction
df['Is_High_Salary'] = df['Salary'] > 70000  # Example feature indicating high salary employees
```
These engineered features, 'Tenure_Years' and 'Is_High_Salary', are intended to improve the predictive power of the model.

### 5. Predictive Modeling
A machine learning model is trained to predict employee churn based on the selected features. The data is split into training and testing sets, and a Random Forest Classifier is used for prediction. The model's performance is then evaluated.
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Define features (X) and target (y)
X = df[['Age', 'Tenure_Years', 'Is_High_Salary']]
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Display the classification report
print(classification_report(y_test, y_pred))
```
The code trains a Random Forest model and evaluates its predictions using a classification report, which includes metrics like precision, recall, and F1-score.

## Tools and Technologies Used

*   Python
*   Google BigQuery
*   pandas
*   scikit-learn

## Key Insights

The predictive model helps identify employees at a higher risk of churning. Understanding key churn drivers, such as limited career growth opportunities or compensation issues, allows the organization to implement targeted retention strategies.

## Conclusion

Analyzing employee churn with data-driven approaches provides organizations with valuable insights to understand and mitigate turnover. This project demonstrates the use of BigQuery and machine learning to extract these insights, ultimately aiding in talent retention.
