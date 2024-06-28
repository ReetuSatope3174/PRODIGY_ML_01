#Import required liabraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import zipfile

# Extract the CSV file from the zip archive (using the correct filename)
with zipfile.ZipFile('house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    zip_ref.extractall() 

# Now read the extracted CSV file
ds= pd.read_csv('train.csv') 
print("File loaded successfully!")
ds

#Define the features to use for training
features=["GrLivArea","BedroomAbvGr", "FullBath"]

#Separate features (x) and target variable (y)
x=ds[features]
y=ds["SalePrice"]

#Split the data into training and testing sets (20% for testing)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Create a linear regression model
model=LinearRegression()

#Train the model on the training data
model.fit(x_train,y_train)

#Predict prices on the testing data
y_pred=model.predict(x_test)

#Evaluate model performance
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-Squared:{r2}')

#Plot actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

#Create a new data point for prediction
new_data = pd.DataFrame({'GrLivArea':[2000],'BedroomAbvGr':[3],'FullBath':[2]})

#Predict price for the new data point
predicted_price=model.predict(new_data)
print(f'Predicted Price:{predicted_price[0]}')
