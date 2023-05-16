import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from PIL import Image
import numpy as np
from keras.models import load_model
import io
from urllib.request import urlopen

def draw_text(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def preprocess_image(image):
    # Resize the image
    image = image.resize((224, 224))
    # Convert the image to an array
    img_array = np.array(image)
    # Scale the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    # Expand the dimensions of the array to create a batch of size 1
    img_batch = np.expand_dims(img_array, axis=0)
    # Return the preprocessed image
    return img_batch

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model('model.h5')
        self.class_names = ['Candi Borobudur', 'Gedung Sate', 'Istana Maimun', 'Jembatan Ampera', 'Monumen Nasional']
        self.class_locations = {
            'Candi Borobudur': {'name': 'Candi Borobudur', 'Latitude': -7.60788, 'Longitude': 110.20367, 'city' : 'Magelang', 'desc': '991.'},
            'Gedung Sate': {'name': 'Gedung Sate', 'Latitude': -6.90249, 'Longitude': 107.61872, 'city' : 'Bandung', 'desc': 'Gon.'},
            'Istana Maimun': {'name': 'Istana Maimun', 'Latitude': 3.5752, 'Longitude': 98.6837, 'city' : 'Medan', 'desc': 'Isge.'},
            'Jembatan Ampera': {'name': 'Jembatan Ampera', 'Latitude': -2.99178, 'Longitude': 104.76354, 'city' : 'Palembang', 'desc': 'Jee.'},
            'Monumen Nasional': {'name': 'Monumen Nasional', 'Latitude': -6.1754, 'Longitude': 106.8272, 'city' : 'Jakarta', 'desc': 'Moia.'},
        }

    def transform(self, frame):
        # Convert the frame to an image
        image = frame.to_ndarray(format='bgr24')
        # Preprocess the image
        img = preprocess_image(Image.fromarray(image))
        # Predict the class probabilities
        probabilities = self.model.predict(img)[0]
        # Get the predicted class index
        predicted_class_idx = np.argmax(probabilities)
        # Get the predicted class name
        predicted_class_name = self.class_names[predicted_class_idx]
        # Get the predicted class location
        predicted_class_location = self.class_locations[predicted_class_name]
        # Draw the predicted class name and location on the image
        draw_text(image, predicted_class_name, (10, 30))
        draw_text(image, predicted_class_location['city'], (10, 60))
        draw_text(image, predicted_class_location['desc'], (10, 90))
        # Return the annotated image
        return image

def main():
    st.title("Landmark Recognition App")
    st.write("Please allow access to your camera to start the app.")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)

if __name__ == '__main__':
    main()
