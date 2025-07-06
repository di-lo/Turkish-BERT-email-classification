import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('haberler.csv')

# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Optionally save them to separate CSV files
train_data.to_csv('test/test_data.csv', index=False)
test_data.to_csv('train/train_data.csv', index=False)

print("done!")