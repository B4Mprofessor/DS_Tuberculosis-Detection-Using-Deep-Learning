import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image # For handling uploaded images in Streamlit
import tensorflow.keras.backend as K

# Define the custom function again (needed for loading the VGG16 model saved with Lambda layer)
def repeat_channels(x):
    return K.repeat_elements(x, 3, axis=-1)

# Load the best saved VGG16 model
# Remember: If the model was saved using the Lambda layer approach, you need custom_objects
@st.cache_resource # Cache the model loading for efficiency in Streamlit
def load_model():
    model = tf.keras.models.load_model('notebooks/models/model_vgg16_best.h5', custom_objects={'repeat_channels': repeat_channels})
    return model

model = load_model()

st.title("Tuberculosis Detection from Chest X-Ray")

uploaded_file = st.file_uploader("Choose a Chest X-Ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray.', use_container_width=True)

    # Preprocess the image to match the training input (1-channel, 256x256)
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Handle potential grayscale/RGB conversion
    # If it's RGB, convert to grayscale. If already grayscale, keep as is.
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Assume RGB, convert to grayscale
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        # Grayscale with channel dim, squeeze it
        img_gray = np.squeeze(img_array, axis=-1)
    elif len(img_array.shape) == 2:
        # Already grayscale
        img_gray = img_array
    else:
        st.error(f"Unexpected image shape: {img_array.shape}")
        st.stop()

    # Resize the image to the required input size (256x256)
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add channel dimension (256, 256) -> (256, 256, 1)
    img_with_channel = np.expand_dims(img_normalized, axis=-1)

    # Add batch dimension (1, 256, 256, 1)
    input_img = np.expand_dims(img_with_channel, axis=0)

    # Make prediction using the model loaded with the Lambda layer
    # The Lambda layer inside the model will convert (1, 256, 256, 1) to (1, 256, 256, 3) internally
    predictions = model.predict(input_img)

    # Get the predicted class (0 for Normal, 1 for TB)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) # Get the confidence of the predicted class

    class_names = ['Normal', 'Tuberculosis'] # Define class names
    result = class_names[predicted_class]

    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence:** {confidence:.4f}")
