import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd


# Configuration
LOOKBACK = 4           # Previous timesteps
INPUT_FEATURES = 9     # Input columns
TARGET_FEATURES = 3    # Target columns
BATCH_SIZE = 12        # Batch size
EPOCHS = 1          # Training epochs

# Generate synthetic data (replace with your data loading)
def generate_data(n_samples=1000):
    t = np.arange(n_samples)
    data = np.zeros((n_samples, INPUT_FEATURES + TARGET_FEATURES))
    
    # Input features (9 columns)
    for i in range(INPUT_FEATURES):
        freq = 0.03 * (i + 1)
        phase = 0.2 * i
        trend = 0.002 * (i + 1) * t
        noise = np.random.normal(0, 0.1, n_samples)
        data[:, i] = np.sin(2 * np.pi * freq * t + phase) + trend + noise
    
    # Target features (3 columns) dependent on inputs
    data[:, INPUT_FEATURES] = 0.5*data[:,0] + 0.3*data[:,2]  # Target 1
    data[:, INPUT_FEATURES+1] = 0.8*data[:,1] - 0.2*data[:,3] # Target 2
    data[:, INPUT_FEATURES+2] = 0.4*data[:,4] + 0.6*data[:,5] # Target 3
    
    return data

# Create dataset with sliding window
def create_dataset(data):
    X, y = [], []
    for i in range(len(data) - LOOKBACK):
        X.append(data[i:i+LOOKBACK, :INPUT_FEATURES])
        y.append(data[i+LOOKBACK, INPUT_FEATURES:])
    return np.array(X), np.array(y)

# Load and preprocess data
# data = generate_data(1200)  # Generate 1200 timesteps
data = np.array(pd.read_excel('data.xlsx'))

# Train-Validate-Test split (60-20-20)
train_size = int(0.6 * len(data))
val_size = int(0.2 * len(data))
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

# Scale data (fit only on training data)
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_inputs = input_scaler.fit_transform(train_data[:, :INPUT_FEATURES])
train_targets = target_scaler.fit_transform(train_data[:, INPUT_FEATURES:])

val_inputs = input_scaler.transform(val_data[:, :INPUT_FEATURES])
val_targets = target_scaler.transform(val_data[:, INPUT_FEATURES:])

test_inputs = input_scaler.transform(test_data[:, :INPUT_FEATURES])
test_targets = target_scaler.transform(test_data[:, INPUT_FEATURES:])

# Create windowed datasets
X_train, y_train = create_dataset(np.column_stack((train_inputs, train_targets)))
X_val, y_val = create_dataset(np.column_stack((val_inputs, val_targets)))
X_test, y_test = create_dataset(np.column_stack((test_inputs, test_targets)))

# Adjust to be divisible by batch size
def adjust_to_batch(X, y, batch_size):
    n_samples = (len(X) // batch_size) * batch_size
    return X[:n_samples], y[:n_samples]

X_train, y_train = adjust_to_batch(X_train, y_train, BATCH_SIZE)
X_val, y_val = adjust_to_batch(X_val, y_val, BATCH_SIZE)
X_test, y_test = adjust_to_batch(X_test, y_test, BATCH_SIZE)

'''
# Build model
model = Sequential([
    LSTM(128, activation='tanh', input_shape=(LOOKBACK, INPUT_FEATURES)), 
    Dropout(0.2), 
    Dense(64, activation='relu'), 
    Dense(TARGET_FEATURES)
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train with validation
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=True
)

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(f"\nTest MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")


model.save('saved_models/LSTM_model_20.keras')
'''

model = tf.keras.models.load_model('saved_models/LSTM_model_20.keras')


for i in range(5):
    # Sample prediction
    sample = X_test[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
    preds = model.predict(sample, batch_size=BATCH_SIZE)
    preds_original = target_scaler.inverse_transform(preds)
    actuals_original = target_scaler.inverse_transform(y_test[BATCH_SIZE * i:BATCH_SIZE * (i + 1)])

    print("\nSample predictions vs actuals:")
    for j in range(12):  # Show first 3 samples
        print(f"\nSample {j+1}:")
        print(f"Predicted: {preds_original[j]}")
        print(f"Actual:    {actuals_original[j]}")

    plt.figure()

    plt.subplot(3, 1, 1)

    plt.plot(actuals_original[:, 0], label='Originals', marker='.', zorder=-10)
    plt.scatter(np.ndarray(len(actuals_original)), actuals_original[:, 0], edgecolors='k', c='#2ca02c', s=64)
    plt.scatter(np.ndarray(len(actuals_original)), preds_original[:, 0], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
    plt.legend()

    plt.subplot(3, 1, 2)

    plt.plot(actuals_original[:, 1], label='Originals', marker='.', zorder=-10)
    plt.scatter(np.ndarray(len(actuals_original)), actuals_original[:, 1], edgecolors='k', c='#2ca02c', s=64)
    plt.scatter(np.ndarray(len(actuals_original)), preds_original[:, 1], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
    plt.legend()

    plt.subplot(3, 1, 3)

    plt.plot(actuals_original[:, 2], label='Originals', marker='.', zorder=-10)
    plt.scatter(np.ndarray(len(actuals_original)), actuals_original[:, 2], edgecolors='k', c='#2ca02c', s=64)
    plt.scatter(np.ndarray(len(actuals_original)), preds_original[:, 2], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
    plt.legend()



# Sample prediction
sample = X_test[-BATCH_SIZE:]
preds = model.predict(sample, batch_size=BATCH_SIZE)
preds_original = target_scaler.inverse_transform(preds)
actuals_original = target_scaler.inverse_transform(y_test[-BATCH_SIZE:])

print("\nSample predictions vs actuals:")
for j in range(12):  # Show first 3 samples
    print(f"\nSample {j+1}:")
    print(f"Predicted: {preds_original[j]}")
    print(f"Actual:    {actuals_original[j]}")

plt.figure()

plt.subplot(3, 1, 1)

plt.plot(actuals_original[:, 0], label='Originals', marker='.', zorder=-10)
plt.scatter(np.ndarray(len(actuals_original)), actuals_original[:, 0], edgecolors='k', c='#2ca02c', s=64)
plt.scatter(np.ndarray(len(actuals_original)), preds_original[:, 0], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
plt.legend()

plt.subplot(3, 1, 2)

plt.plot(actuals_original[:, 1], label='Originals', marker='.', zorder=-10)
plt.scatter(np.ndarray(len(actuals_original)), actuals_original[:, 1], edgecolors='k', c='#2ca02c', s=64)
plt.scatter(np.ndarray(len(actuals_original)), preds_original[:, 1], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
plt.legend()

plt.subplot(3, 1, 3)

plt.plot(actuals_original[:, 2], label='Originals', marker='.', zorder=-10)
plt.scatter(np.ndarray(len(actuals_original)), actuals_original[:, 2], edgecolors='k', c='#2ca02c', s=64)
plt.scatter(np.ndarray(len(actuals_original)), preds_original[:, 2], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
plt.legend()

plt.show()