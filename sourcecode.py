import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the gold price dataset
df = pd.read_csv('FINAL_USO 1.csv')  # Replace 'your_uploaded_filename.csv' with the actual filename

# Select the relevant columns for prediction
selected_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume',
                    'SP_open', 'SP_high', 'SP_close', 'SP_Ajclose', 'SP_volume',
                    'DJ_open', 'DJ_high', 'DJ_close', 'DJ_volume',
                    'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume',
                    'EU_Price', 'EU_open', 'EU_high', 'EU_low', 'EU_Trend',
                    'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend']

df = df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=1, cbar=True, cbar_kws={'shrink': 0.8})
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, ha='center', fontsize=8)
plt.title('Correlation Matrix', fontsize=12)
plt.tight_layout()
plt.show()

# Split the dataset into features (X) and target variable (y)
X = df.drop('Adj Close', axis=1)  # Assuming 'Adj Close' is the column containing the target variable
y = df['Adj Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Create a scatter plot of actual vs predicted prices with trendline
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, color='blue', marker='o', s=30, label='Actual vs Predicted Prices')
plt.plot(np.linspace(0, np.max(y_test)), np.linspace(0, np.max(y_test)), color='red', label='Trendline')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Gold Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
