import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
with open('gesture_dataset.pickle', 'rb') as f:  # Updated filename
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])  # Labels should be D, W, B

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('gesture_model.p', 'wb') as f:  # New filename for clarity
    pickle.dump({'model': model}, f)

print("Model trained and saved successfully!")