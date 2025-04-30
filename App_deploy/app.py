import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

model = keras.models.load_model('clothes_model.keras')

st.title('Clothes Image Identifier')
st.write('I bet I can guess what kind of clothes are pictured in an uploaded image')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    if st.button('Predict'):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.success(f"Predicted: {predicted_class} ({confidence*100:.2f}% confidence)")
