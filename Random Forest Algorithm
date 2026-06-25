# import the libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine 
import urllib

## connect SQL Server to Python

# format the parameters first
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=.\SQLEXPRESS;"
    "DATABASE=Electronic_Sales;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

# 2. Pass it into the create_engine URL wrapper
connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
engine = create_engine(connection_string)


# fetch data from SQL Database

sql_query = "SELECT * FROM [dbo].[vw_Churned]"
data = pd.read_sql(sql_query, con=engine)

# display the first 5 rows of the data
print(data.head(5))

# Check the datatypes
data.info()

# convert Order_Date from String to Date
data['Order_Date'] = pd.to_datetime(data['Order_Date'])

# columns to encode
label_encoders = {}
cols_to_encode = ['Gender', 'Loyalty_Member', 'Product_Type', 'Payment_Method']

for column in cols_to_encode:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# split data into Features and Targets
X = data.drop(columns = ['Customer_ID', 'Last_Order_Date', 'Days_Since_Last_Purchase', 'Customer_Status', 'Customer_Status_Binary', 'Order_Date'])
y = data['Customer_Status']

# split data into Training and Training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators = 200, class_weight = 'balanced', random_state = 42)
rf_model.fit(X_train, y_train)

## Evaluate Model

# make predictions
y_pred = rf_model.predict(X_test)

# evaluate the model
print("confusion_matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# feature selection using feature importances_
importances = rf_model.feature_importances_
indices = np.argsort(importances) [:: -1]

# plot the feature importances in a chart
plt.figure(figsize = (5,2))
sns.barplot(x = importances[indices], y = X.columns[indices])
plt.title('Feature Importances')
plt.xlabel('Relative Importances')
plt.ylabel('Feature Names')
plt.show()

# predict on new data
sql_query_new = "SELECT * FROM [dbo].[vw_stayed]"
new_data = pd.read_sql(sql_query_new, con = engine)
new_data

# retain the original dataframe to preserve the unencoded columns
original_data = new_data.copy()

# retain Customer_ID from the existing series
Customer_ids = new_data['Customer_ID']

# Drop columns that won't be used for prediction 
columns_to_drop = ['Customer_ID', 'Last_Order_Date', 'Days_Since_Last_Purchase', 'Customer_Status', 'Customer_Status_Binary', 'Order_Date']
new_data = new_data.drop(columns=columns_to_drop)

# Encode categorical columns safely using the saved dictionary
cols_to_encode = ['Gender', 'Loyalty_Member', 'Product_Type', 'Payment_Method']

for column in cols_to_encode:
    new_data[column] = label_encoders[column].transform(new_data[column])

# 3. Make predictions
new_predictions = rf_model.predict(new_data)

# add predictions to the original dataframe
original_data['Customer_Status_Predicted'] = new_predictions

# filter the dataframe to include on;y records predicted as 'churned'
original_data = original_data[original_data['Customer_Status_Predicted'] ==1]

# save the result: push it back to SQL Server
original_data.to_sql('original_data', con = engine, if_exists = 'append', index = False)

print(f'successfully exported records to database table')









