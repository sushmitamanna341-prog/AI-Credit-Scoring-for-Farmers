from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import joblib
import numpy as np
import pandas as pd
import os

# Load models and encoders
random_forest_model = joblib.load('random_forest_model.pkl')
clf_attack = joblib.load('attack_type_classifier.pkl')
scaler = joblib.load('scaler.pkl')

encoders = {
    'crop_type': joblib.load('crop_type_encoder.pkl'),
    'location_zone': joblib.load('location_zone_encoder.pkl'),
    'credit_score': joblib.load('credit_score_encoder.pkl'),
    'image_path': joblib.load('image_path_encoder.pkl'),
}

# Load cleaned dataset with relative image paths
df = pd.read_csv("crop_dataset_cleaned.csv")

# ==== GUI ====
root = Tk()
root.title("Farmer Credit Score Predictor")
root.geometry("1000x700")

bg_image_path = "background.jpg"
if not os.path.exists(bg_image_path):
    messagebox.showerror("Error", f"Background image not found: {bg_image_path}")
    root.destroy()
    exit()

bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((1000, 700))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

title_label = Label(root, text="Farmer's Crop Credit Score Prediction",
                    font=("Helvetica", 20, "bold"), bg="#ffffff", fg="#2c3e50")
title_label.place(relx=0.5, rely=0.08, anchor=CENTER)

form_frame = Frame(root, bg="white", bd=4)
form_frame.place(relx=0.3, rely=0.55, anchor=CENTER)

# ==== Input Fields ====
Label(form_frame, text="Crop Type:", bg="white").grid(row=0, column=0, padx=10, pady=5, sticky=W)
crop_type_entry = Entry(form_frame)
crop_type_entry.grid(row=0, column=1)

Label(form_frame, text="Location Zone:", bg="white").grid(row=1, column=0, padx=10, pady=5, sticky=W)
location_entry = Entry(form_frame)
location_entry.grid(row=1, column=1)

Label(form_frame, text="Precipitation (mm):", bg="white").grid(row=2, column=0, padx=10, pady=5, sticky=W)
precip_entry = Entry(form_frame)
precip_entry.grid(row=2, column=1)

Label(form_frame, text="Avg Temperature (°C):", bg="white").grid(row=3, column=0, padx=10, pady=5, sticky=W)
temp_entry = Entry(form_frame)
temp_entry.grid(row=3, column=1)

Label(form_frame, text="Soil Moisture (%):", bg="white").grid(row=4, column=0, padx=10, pady=5, sticky=W)
moisture_entry = Entry(form_frame)
moisture_entry.grid(row=4, column=1)

Label(form_frame, text="Image Index (0-999):", bg="white").grid(row=5, column=0, padx=10, pady=5, sticky=W)
img_index_entry = Entry(form_frame)
img_index_entry.grid(row=5, column=1)

Label(form_frame, text="Yield 1:", bg="white").grid(row=6, column=0, padx=10, pady=5, sticky=W)
yield1_entry = Entry(form_frame)
yield1_entry.grid(row=6, column=1)

Label(form_frame, text="Yield 2:", bg="white").grid(row=7, column=0, padx=10, pady=5, sticky=W)
yield2_entry = Entry(form_frame)
yield2_entry.grid(row=7, column=1)

Label(form_frame, text="Yield 3:", bg="white").grid(row=8, column=0, padx=10, pady=5, sticky=W)
yield3_entry = Entry(form_frame)
yield3_entry.grid(row=8, column=1)

# Image and score display
image_label = Label(root)
image_label.place(relx=0.7, rely=0.45, anchor=CENTER)

score_label = Label(root, text="", font=("Helvetica", 14, "bold"), bg="white", fg="blue")
score_label.place(relx=0.7, rely=0.72, anchor=CENTER)

# ==== Predict Function ====
def predict_credit_score():
    try:
        crop = crop_type_entry.get()
        zone = location_entry.get()
        precipitation = round(float(precip_entry.get()), 2)
        temperature = round(float(temp_entry.get()), 2)
        soil = round(float(moisture_entry.get()), 2)
        index = round(float(img_index_entry.get()))
        yield_1 = round(float(yield1_entry.get()), 2)
        yield_2 = round(float(yield2_entry.get()), 2)
        yield_3 = round(float(yield3_entry.get()), 2)

        if not (0 <= precipitation <= 500):
            messagebox.showerror("Input Error", "Precipitation must be between 0 and 500 mm.")
            return
        if not (-10 <= temperature <= 60):
            messagebox.showerror("Input Error", "Temperature must be between -10°C and 60°C.")
            return
        if not (0 <= soil <= 100):
            messagebox.showerror("Input Error", "Soil moisture must be between 0% and 100%.")
            return
        if index < 0 or index >= len(df):
            messagebox.showerror("Error", "Image index out of range.")
            return

        image_path = df.loc[index, 'Image Path'].strip()

        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image not found: {image_path}")
            return

        crop_enc = encoders['crop_type'].transform([crop])[0]
        zone_enc = encoders['location_zone'].transform([zone])[0]
        image_enc = encoders['image_path'].transform([image_path])[0]

        features = np.array([[crop_enc, zone_enc, precipitation, temperature, soil, yield_1, yield_2, yield_3]])
        scaled = scaler.transform(features)
        pred = clf_attack.predict(scaled)
        label = encoders['credit_score'].inverse_transform(pred)

        score_label.config(text=f"Predicted Credit Score: {label[0]}")
        messagebox.showinfo("Prediction", f"Predicted Credit Score: {label[0]}")

        img = Image.open(image_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ==== Predict Button ====
predict_btn = Button(form_frame, text="Predict Credit Score", command=predict_credit_score, bg="green", fg="white")
predict_btn.grid(row=9, columnspan=2, pady=10)

root.mainloop()
