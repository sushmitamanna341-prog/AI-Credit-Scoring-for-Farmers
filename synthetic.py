import os
import random
import pandas as pd
import numpy as np

# Define crop types and location zones
crop_types = ['Wheat', 'Rice', 'Maize', 'Barley', 'Cotton']
location_zones = ['North', 'South', 'East', 'West']

# Define the path to the images folder
image_folder = "C:/Users/Edify/Desktop/chan-projects/working/farmer/permanent_crop"

# Get all image filenames
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])

# Limit the number of images to 1000
image_files = image_files[:1000]

# Generate synthetic data
data = []

for image_file in image_files:
    # Assign random features
    crop_type = random.choice(crop_types)
    location_zone = random.choice(location_zones)
    soil_moisture = round(random.uniform(20.0, 80.0), 2)
    precipitation = round(random.uniform(100.0, 500.0), 2)
    avg_temperature = round(random.uniform(15.0, 35.0), 2)
    
    # Random credit score (numeric)
    credit_score = random.randint(650, 850)

    # Add the record to the dataset
    data.append([
        crop_type, location_zone, soil_moisture, precipitation, avg_temperature, 
        os.path.join(image_folder, image_file), credit_score
    ])

# Create a DataFrame
df = pd.DataFrame(data, columns=[
    'Crop Type', 'Location Zone', 'Soil Moisture (%)', 'Precipitation (mm)', 
    'Avg Temperature (°C)', 'Image Path', 'Credit Score'
])

# Save the synthetic dataset to CSV
df.to_csv('synthetic_crop_dataset.csv', index=False)

print(f"Generated synthetic dataset with {len(df)} rows.")
