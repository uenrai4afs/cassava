import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time
import base64

# Set page title and header
st.set_page_config(page_title="Cassava Disease/Pest Detection App")
st.header('Cassava Disease/Pest Detection App')

# Function to set background image
def set_background(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background('cassava.png')

# Load the TFLite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Make prediction
def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    probabilities = np.array(output_data[0])
    return probabilities

# Main function
def main():
    # Load model
    model_path = "cassava.tflite"
    interpreter = load_model(model_path)

    # File uploader
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=False)
        
        # Detect button
        if st.button("Detect"):
            with st.spinner('Model working....'):
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Make prediction
                probabilities = predict(interpreter, processed_image)
                
                # Process results
                labels = ["bacterial blight", "brown spot", "green mite", "healthy", "mosaic", "non type"]
                highest_prob_index = np.argmax(probabilities)
                result_category = labels[highest_prob_index]
                confidence = 100 * np.max(probabilities)
                
                # Display results
                st.success('Results')
                st.write(f"Category: {result_category}")
                st.write(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
