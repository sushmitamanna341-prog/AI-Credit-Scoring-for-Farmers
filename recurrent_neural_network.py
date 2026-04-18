import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ===============================
# Load and preprocess dataset
# ===============================
df = pd.read_csv("crop_dataset.csv")

# Encode categorical features
df['crop_type'] = LabelEncoder().fit_transform(df['crop_type'])
df['location_zone'] = LabelEncoder().fit_transform(df['location_zone'])

# Drop image path column if present (prevents string->float error)
if 'image_path' in df.columns:
    df = df.drop(columns=['image_path'])

# Load images
image_folder = 'permanent_crop'
image_size = (64, 64)

image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])
images = [img_to_array(load_img(path, target_size=image_size)) / 255.0 for path in image_paths]
images = np.array(images)

# Prepare tabular data (excluding 'credit_score' and any non-numeric columns)
X_tabular = df.drop(['credit_score', 'index_of_image'], axis=1)
y = df['credit_score']

# Normalize tabular features
scaler = StandardScaler()
X_tabular_scaled = scaler.fit_transform(X_tabular)

# Match lengths between image and tabular data
min_len = min(len(images), len(X_tabular_scaled), len(y))
images = images[:min_len]
X_tabular_scaled = X_tabular_scaled[:min_len]
y = y[:min_len]

# Flatten images and combine with tabular features
X_img_flat = images.reshape(images.shape[0], -1)
X_combined = np.hstack([X_img_flat, X_tabular_scaled])

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y)

# Shuffle data
X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

# Reshape for LSTM: (samples, time_steps, features)
time_steps = 10
features_per_step = X_resampled.shape[1] // time_steps
X_rnn = X_resampled[:, :time_steps * features_per_step].reshape(-1, time_steps, features_per_step)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_rnn, y_resampled, test_size=0.2, random_state=42)

# ===============================
# Build LSTM model
# ===============================
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, features_per_step)))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 4-class classification

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===============================
# Train model
# ===============================
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# ===============================
# Evaluate model
# ===============================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n✅ Recurrent Neural Network Accuracy: {acc * 100:.2f}%")
print(f"✅ F1 Score: {f1:.4f}")

# Save model
model.save('rnn_model.h5')
print("✅ RNN model saved as 'rnn_model.h5'")
