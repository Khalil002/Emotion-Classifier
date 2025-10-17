import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/imageclassifier.keras")

IMG_SIZE = (256, 256)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def classify_image():
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*"),
        ]
    )
    if not file_path:
        return

    img_array, pil_image = preprocess_image(file_path)

    pred = model.predict(img_array)[0][0]
    label_text = "Sad" if pred > 0.5 else "Happy"
    confidence = pred if pred > 0.5 else 1 - pred
    confidence_percentage = confidence * 100

    result_label.config(
        text=f"Prediction: {label_text} ({confidence_percentage:.2f}% confident)"
    )

    pil_image = pil_image.resize((200, 200))
    tk_image = ImageTk.PhotoImage(pil_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image

root = tk.Tk()
root.title("Emotion Classifier")

upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="Prediction: ")
result_label.pack(pady=10)

root.mainloop()
