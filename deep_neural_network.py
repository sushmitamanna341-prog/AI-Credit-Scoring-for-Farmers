import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# === Load dataset ===
df = pd.read_csv("crop_dataset_cleaned.csv")

# === Encode categorical columns ===
encoders = {}
for col in ['crop_type', 'location_zone']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# === Encode target column (credit_score) ===
le_score = LabelEncoder()
df['credit_score'] = le_score.fit_transform(df['credit_score'])
encoders['credit_score'] = le_score

# === Encode Image Path as a categorical feature ===
le_image = LabelEncoder()
df['Image Path'] = le_image.fit_transform(df['Image Path'])
encoders['image_path'] = le_image

# === Features and target ===
features = ['crop_type', 'location_zone', 'precipitation', 'avg_temperature', 'soil_moisture', 'Image Path','Yield_1','Yield_2','Yield_3']
X = df[features]
y = df['credit_score']

# === Remove rare classes (<2 samples) ===
class_counts = y.value_counts()
classes_to_remove = class_counts[class_counts < 2].index
df_filtered = df[~df['credit_score'].isin(classes_to_remove)]
X_filtered = df_filtered[features]
y_filtered = df_filtered['credit_score']

# === Scale numerical features ===
scaler = StandardScaler()
X_scaled_filtered = scaler.fit_transform(X_filtered.drop(columns=['Image Path']))

# === Handle imbalance using oversampling ===
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled_filtered, y_filtered)

# === Prepare labels for DNN (one-hot encode) ===
y_res_cat = to_categorical(y_resampled)

# === Define DNN model ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_resampled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_res_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Early stopping callback ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === Train the model ===
model.fit(X_resampled, y_res_cat, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

# === Evaluation ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled_filtered, y_filtered, test_size=0.2, stratify=y_filtered)

y_test_cat = to_categorical(y_train, num_classes=y_filtered.max() + 1)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Map back to actual class labels
unique_classes = np.unique(y_filtered)

print("\nDNN Classification Report:\n")
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=[str(cls) for cls in unique_classes], zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOverall Accuracy: {accuracy:.4f}")
print(f"Overall Weighted F1 Score: {f1:.4f}")

# === Save model and encoders ===
model.save('dnn_classifier.h5')
joblib.dump(scaler, 'scaler_dnn.pkl')

for col, enc in encoders.items():
    joblib.dump(enc, f'{col.lower().replace(" ", "_")}_encoder.pkl')

print("\nDNN model and encoders saved successfully.")
