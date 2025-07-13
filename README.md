# Predictive-maintenance-for-fleet-tyre-
#Import packages 
Import pandas as pd 
Import numpy as np 
From datatime import datetime 
From sklearn.model_section import train_test_split 
From sklearn,ensemble import RandomForestClassifier , RandomForestRegressor 
From sklearn.metrics import accuracy_score,mean_squared_error 
#Load dataset 
Data=pd.read_csv(“realistic_fleet_tyre_data.csv”) 
#Define average usage per day(km/day) 
Avg_km_per_day=100#preprocessing : calculate days in use and total kilometers travelled 
Data['Days In Use'] = (datetime.now() - pd.to_datetime(data['Installation Date'])).dt.days  
data['Total Kilometers'] = data['Days In Use'] * avg_km_per_day 
#calculate condition score to use as target for classification model 
def calculate_condition_score(psi, depth, temp): psi_factor = 1 if 90 <= psi <= 110 else 0  
depth_factor = 1 if depth > 6 else 0.5 if 3 <= depth <= 6 else 0 
temp_factor = 1 if 20 <= temp <= 35 else 0 return psi_factor + depth_factor + temp_factor 
# Apply condition score and assign condition labels 
data['ConditionScore']=data.apply(lambda row:  
calculate_condition_score(row[‘PSI’],row[‘TyreDepth(mm)’] , row['Temperature (°C)']),  
axis=1) data['Condition'] = data['Condition Score'].apply(lambda score: "Good" if score  
== 3 else "Average" if score >= 2 else "Bad") 
# Add 'Remaining Kilometers' if it doesn’t exist by calculating based on expected lifetime  
if 'Remaining Kilometers' not in data.columns: 
expected_lifetime = data['Condition'].map({'Good': 100000, 'Average': 75000, 'Bad':  
50000}) 
data['Remaining Kilometers'] = expected_lifetime - data['Total Kilometers'] 
# Define features and targets for model training: 
X = data[['PSI', 'Tyre Depth (mm)', 'Temperature (°C)', 'Total Kilometers']] 
y_condition = data['Condition'] 
y_remaining_km = data['Remaining Kilometers'] 
# Split data into training and test sets: 
X_train, X_test, y_condition_train, y_condition_test = train_test_split(X, y_condition,  
test_size=0.2, random_state=42) 
_, _, y_remaining_km_train, y_remaining_km_test = train_test_split(X, y_remaining_km,  
test_size=0.2, random_state=42) 
# Train a classifier for tire condition: 
Condition_model=RandomForestClassifier(random_state=42) 
condition_model.fit(X_train, y_condition_train) 
# Train a regressor for remaining kilometers 
remaining_km_model=RandomForestRegressor(random_state=42) 
remaining_km_model.fit(X_train, y_remaining_km_train) 
# Model evaluation:condition_preds = condition_model.predict(X_test) 
remaining_km_preds = remaining_km_model.predict(X_test)

print(f"Condition Model Accuracy: {accuracy_score(y_condition_test, 

condition_preds)}")

print(f"Remaining KM Model RMSE: 

{np.sqrt(mean_squared_error(y_remaining_km_test, remaining_km_preds))}")

# Evaluate each tire and generate the formatted output:

results = []

for index, row in data.iterrows():

# Create a DataFrame for the current row with proper column names for prediction:

input_data = pd.DataFrame({

 'PSI': [row['PSI']],

 'Tyre Depth (mm)': [row['Tyre Depth (mm)']],

 'Temperature (°C)': [row['Temperature (°C)']],

 'Total Kilometers': [row['Total Kilometers']]

 })

 

 # Predict condition and remaining kilometers

 condition_pred = condition_model.predict(input_data)[0]

 remaining_km_pred = remaining_km_model.predict(input_data)[0]

 

# Format the values to one decimal place

 psi_formatted = f"{row['PSI']:.1f}"

 depth_formatted = f"{row['Tyre Depth (mm)']:.1f}"

 temp_formatted = f"{row['Temperature (°C)']:.1f}"

 

# Logic for determining replacement or remaining km

 if condition_pred == "Bad" or remaining_km_pred <= 0:

 results.append(f"{row['Tyre ID']} is in {condition_pred} condition with 

[{psi_formatted} PSI, {depth_formatted} mm, {temp_formatted}°C], so it needs to be 

replaced immediately.")

 else:

 results.append(f"{row['Tyre ID']} is in {condition_pred} condition with 

[{psi_formatted} PSI, {depth_formatted} mm, {temp_formatted}°C], so no need to replace and can travel {remaining_km_pred:.0f} km more.")

# Print the results
for result in results:
print(result)
