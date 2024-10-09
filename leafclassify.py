import streamlit as st
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from keras.models import *
from keras import preprocessing
import time
import base64
## this is part of web app

## ----------------------------------------------- x -----------------------------------------x-------------------------x------------------##


# fig = plt.figure()
st.markdown("<h1 style='color: black;'>Cassava Disease/Pest Prediction</h1>", unsafe_allow_html=True)

#st.markdown("Prediction Platform")
def set_background(main_bg):  # local image
    # set bg name
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


set_background('cassava.png')
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=False)
    class_btn = st.button("Detect")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success('Results')
                #st.write(predictions)
                st.markdown(predictions, unsafe_allow_html=True)


## This code is for saved model in format as H5 file


# def predict(image):
#     classifier_model = "HD_Model.h5"

#     model = load_model(classifier_model)

#     test_image = image.resize((200,200))
#     test_image = preprocessing.image.img_to_array(test_image)
#     test_image = test_image / 255.0
#     test_image = np.expand_dims(test_image, axis=0)
#     class_names = {0 : 'healthy', 1 :'diseased'}

#     predictions = model.predict(test_image)
#     scores = tf.nn.softmax(predictions[0])
#     scores = scores.numpy()


#     result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
#     return result


## -----------------------------------------------------x---------------------------------------x--------------------------------------------##


## this code for format tflite file
def predict(image):
    #model = "leaves_model.tflite"
    model="cassava.tflite"
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = np.array(image.resize((224, 224)), dtype=np.float32)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])

    labels = {0: "bacterial blight", 1: "brown spot", 2: "green mite", 3: "healthy", 4: "mosaic", 5: "non type" }
    label_new=["bacterial blight", "brown spot", "green mite", "healthy", "mosaic", "non type" ]


    label_to_probabilities = []
    print(probabilities)

    for i, probability in enumerate(probabilities):
        label_to_probabilities.append([labels[i], float(probability)])

    sorted(label_to_probabilities, key=lambda element: element[1])

    #result = {'healthy': 0, 'diseased': 0}
    #result = {'leaf Blight': 0, 'brown spot': 0, 'greenmite': 0, 'healthy': 0, 'mosaic': 0}
    #result = f"{label_to_probabilities[np.argmax(probability)][0]} with a {(100 * np.max(probabilities)).round(2)} % confidence."
    #result=f"{} with a {}"
    high=np.argmax(probabilities)
    result_1=label_new[high]
    confidence=100 * np.max(probabilities)
    #result="Category:"+ "  "+str(result_1) +"     "+ "\n Confidence: "+ " "+ str(confidence)+ "%" 
    result = f"""
    <span style='color: black;'>
        Category: <strong>{result_1}</strong><br>
        Confidence: <strong>{confidence:.2f}%</strong>
    </span>
    """

    return result


if __name__ == "__main__":
    main()
