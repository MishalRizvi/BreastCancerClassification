import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.datasets


# Set random seeds for reproducibility
tf.random.set_seed(3)
np.random.seed(3)

# Load the dataset
dataset = sklearn.datasets.load_breast_cancer()
data = pd.DataFrame(dataset.data, columns = dataset.feature_names)

print("number of samples in the dataset", data.shape[0])

#Add the target column to the dataframe
data['label'] = dataset.target

#First 5 rows of the dataset
# print(data.head())

#Information about the dataset
# print(data.info())

#Statistical summary of numerical columns 
# print(data.describe())

#Separate features (X) and target (Y)
X = data.drop(['label'], axis=1)
Y = data['label']
#Split data into training and testing sets - training set 80% and testing set 20%
#random state ensures same split of data every time code is run, ensures reproducibility 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.head())
print(Y_train.head())
print(X_test.head())
print(Y_test.head())

#Scale the features 
#fit_transform() computes the mean and std to be used for scaling of test data 
#transform() uses the learned mean and std from train data to scale the test data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create the model 
#Each layer is a dense layer with 16, 8, and 1 neurons respectively 
#Each neuron recieves inputs from the previous layer, applies weights to each input, adds a bias term and applies an activation function 
#Layer 1 learns lower-level patterns in data 
#Layer 2 learns higher-level combinations of patterns 
#Layer 3 combines all learned patterns to make a final prediction 

# model = keras.Sequential([
#     keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     keras.layers.Dense(8, activation='relu'), #ReLu allows for non-linear relationships 
#     keras.layers.Dense(1, activation='sigmoid') #Sigmoid perfect for binary classification 
# ])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

#Non-linear means for example Y = X^2
#A tumor's malignancy might not increase proportionally with size

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Training process:
#1. Forward pass: Input -> Layer 1 -> Layer 2 -> Output (Prediction is made)
#2. Backward pass 1: Compute loss (difference between prediction and actual target)
#3. Backward pass 2: Adjust weights to minimize loss using Adam optimiser 
#4. Repeat for multiple epochs to minimise cross-entropy loss 


#Fit the model to the training data 
history = model.fit(X_train_scaled, Y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
# Visualize training results
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


#Evaluate the model on the test data 
test_loss, test_accuracy = model.evaluate(X_test_scaled, Y_test, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

Y_predicted_values = model.predict(X_test_scaled)
Y_predicated_labels = (Y_predicted_values > 0.5).astype(int) #Converts probabilities to binary labels 

def predict_benign_or_malignant(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Single sample without reshape
    # input_data = [5.1, 3.5, 1.4, 0.2]  # Shape: (4,)

    # Same sample with reshape
    # input_data_reshaped = [[5.1, 3.5, 1.4, 0.2]]  # Shape: (1, 4)

    input_data_std = scaler.transform(input_data_reshaped)
    prediction = model.predict(input_data_std)
    # prediction_label = (prediction > 0.5).astype(int)
    prediction_label = [np.argmax(prediction)]
    # print("prediction_label", prediction_label)
    print("prediction_label", prediction_label[0])
    if prediction_label[0] == 0:
        return 'The tumor is Malignant'
    else:
        return 'The tumor is Benign'


correct_predictions = 0
for i in range(X_test.shape[0]):
    sample_data = X_test.iloc[i].values
    print(f"\nSample {i+1}:")
    predicted_value = predict_benign_or_malignant(sample_data)
    print("Prediction:", predicted_value)
    print("Actual value:", "Malignant" if Y_test.iloc[i] == 0 else "Benign")
    print("-" * 50)
    if (predicted_value == "The tumor is Malignant" and Y_test.iloc[i] == 0):
        correct_predictions += 1
    elif (predicted_value == "The tumor is Benign" and Y_test.iloc[i] == 1):
        correct_predictions += 1

print(f"Accuracy: {correct_predictions / X_test.shape[0] * 100:.2f}%")

"""1 --> Benign

0 --> Malignant
"""




