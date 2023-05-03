import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static
import cv2

# Define the class names
nama_class = ['Candi Borobudur', 'Gedung Sate', 'Istana Maimun', 'Jembatan Ampera', 'Monumen Nasional']

# Define the locations of each class
class_locations = {
    'Candi Borobudur': {'name': 'Candi Borobudur', 'Latitude': -7.60788, 'Longitude': 110.20367, 'city' : 'Magelang',
                        'desc': '991.'},
	'Gedung Sate': {'name': 'Gedung Sate', 'Latitude': -6.90249, 'Longitude': 107.61872, 'city' : 'Bandung',
                    'desc': 'Gon.'},
	'Istana Maimun': {'name': 'Istana Maimun', 'Latitude': 3.5752, 'Longitude': 98.6837, 'city' : 'Medan',
                      'desc': 'Isge.'},
    'Jembatan Ampera': {'name': 'Jembatan Ampera', 'Latitude': -2.99178, 'Longitude': 104.76354, 'city' : 'Palembang',
                        'desc': 'Jee.'},
	'Monumen Nasional': {'name': 'Monumen Nasional', 'Latitude': -6.1754, 'Longitude': 106.8272, 'city' : 'Jakarta',
                         'desc': 'Moia.'},
}

# Load the model
model = load_model('model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define the prediction function
def predict(image):
    try:
        # Preprocess the image
        img = preprocess_image(image)
        # Predict the class probabilities
        probabilities = model.predict(img)[0]
        # Get the predicted class index
        predicted_class_idx = np.argmax(probabilities)
        # Get the predicted class name
        predicted_class = nama_class[predicted_class_idx]
        # Get the probability of the predicted class
        predicted_prob = probabilities[predicted_class_idx]
        # Convert the probability to a percentage
        predicted_prob_pct = round(predicted_prob * 100, 2)
        # Convert the probabilities to percentages
        probabilities_pct = [round(prob * 100, 2) for prob in probabilities]
        # Return the predicted class name and probabilities
        return (predicted_class, predicted_prob_pct, probabilities_pct)
    except:
        # Return None if the image cannot be processed
        return None

# Set up the Streamlit app
st.set_page_config(page_title="Image Classification", page_icon=":smiley:")
st.title("Klasifikasi Landmark")
st.write("Unggah gambar atau gunakan kamera untuk mengklasifikasikan landmark ke dalam salah satu kelas berikut:")
st.write(nama_class)

# Add a map to the app
geolocator = Nominatim(user_agent="app")
location = geolocator.geocode("Indonesia") # Initial location
m = folium.Map(location=[location.latitude, location.longitude], zoom_start=5)

# Add a camera button to the app
use_camera = st.button("Gunakan Kamera")

if use_camera:
    # Create a VideoCapture object to capture images from camera
    cap = cv2.VideoCapture(0)
    # Capture an image from the camera
    ret, frame = cap.read()
    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Show the captured image
    st.image(image, caption='Gambar dari Kamera', use_column_width=True)
    # Make a prediction
    prediction = predict(image)
    if prediction is not None:
        predicted_class, predicted_prob, probabilities = prediction
        # Show the predicted class and probability
        st.write("Predicted class:", predicted_class)
        st.write("Probability:", predicted_prob, "%")
        # Show the probabilities for each class
        for class_name, prob in zip(nama_class, probabilities):
            st.write(class_name, ":", prob, "%")
        # Get the location of the predicted class
        class_location = class_locations[predicted_class]
        # Add a marker to the map
        folium.Marker(
            location=[class_location['Latitude'], class_location['Longitude']],
            popup=class_location['name'],
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        # Zoom to the location
        m.fit_bounds([[class_location['Latitude'], class_location['Longitude']]])
        # Show the class location
        st.write("Address:", class_location)
        # Update the map
        folium_static(m, width=700, height=500)

else:
    # Add a file uploader to the app
    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        # Show the image
        st.image(image, caption='Unggah Gambar', use_column_width=True)
        # Make a prediction
        prediction = predict(image)
        if prediction is not None:
            predicted_class, predicted_prob, probabilities = prediction
            # Show the predicted class and probability
            st.write("Predicted class:", predicted_class)
            st.write("Probability:", predicted_prob, "%")
            # Show the probabilities for each class
            for class_name, prob in zip(nama_class, probabilities):
                st.write(class_name, ":", prob, "%")
            # Get the location of the predicted class
            class_location = class_locations[predicted_class]
            # Add a marker to the map
            folium.Marker(
                location=[class_location['Latitude'], class_location['Longitude']],
                popup=class_location['name'],
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            # Zoom to the location
            m.fit_bounds([[class_location['Latitude'], class_location['Longitude']]])
            # Show the class location
            st.write("Address:", class_location)
            # Update the map
            folium_static(m, width=700, height=500)
