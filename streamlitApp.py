import streamlit as st
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid
import pickle
from streamlit_option_menu import option_menu 
# loading the saved models

drawing_model = load_model("./keras_model.h5", compile=False)
image_model = load_model("./keras_model.h5", compile=False)

# sidebar navigation
with st.sidebar:
    
    selected = option_menu('Parkinson Detection - Spiral model', 
                           ['Dynamic Spiral Model',
                            'Visual Input Spiral Model'],
                           icons=['pen','camera'],
                           default_index=0)
    

    st.subheader("What is Parkinson??")
    st.write("Parkinson's disease, a neurodegenerative condition impacting motor functions, manifests through symptoms such as tremors, stiffness, and compromised movement. The study highlighted in the provided article investigates the utilization of spiral and wave sketch images to formulate a robust algorithm for detecting Parkinson's disease. Parkonix utilizes these sketch images as training data for an AI model, attaining a noteworthy accuracy rate of 80%.")
    link_text = "Discerning Various Phases of Parkinson's Disease Through a Composite Index of Speed and Pen-Pressure in Sketching a Spiral."
    link_url = "https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full"
    st.markdown(f"[{link_text}]({link_url})")

if (selected == 'Dynamic Spiral Model'): 
    st.header("Detecting Parkinson's Disease - Dynamic Spiral Model")
    with st.sidebar:
        # Specify canvas parameters in application
        drawing_mode = "freedraw"

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke colour : ")
        bg_color = st.sidebar.color_picker("Background colour : ", "#eee")

        realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Split the layout into two columns
    col1, col2 = st.columns(2)

    # Define the canvas size
    canvas_size = 345

    with col1:
        # Create a canvas component
        st.subheader("Drawable Interface")
        canvas_image = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            width=canvas_size,
            height=canvas_size,
            update_streamlit=realtime_update,
            drawing_mode=drawing_mode,
            key="canvas",
        )

    with col2:
        st.subheader("Overview")
        if canvas_image.image_data is not None:
            # Get the numpy array (4-channel RGBA 100,100,4)
            input_numpy_array = np.array(canvas_image.image_data)
            # Get the RGBA PIL image
            input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
            st.image(input_image, use_column_width=True)

    def generate_user_input_filename():
        unique_id = uuid.uuid4().hex
        filename = f"user_input_{unique_id}.png"
        return filename

    def predict_parkinsons(img_path):
        best_model = load_model("./keras_model.h5", compile=False)

        # Load the labels
        class_names = open("labels.txt", "r").readlines()

        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Get the numpy array (4-channel RGBA 100,100,4)
        input_numpy_array = np.array(img_path.image_data)

        # Get the RGBA PIL image
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")

        # Generate a unique filename for the user input
        user_input_filename = generate_user_input_filename()

        # Save the image with the generated filename
        input_image.save(user_input_filename)
        print("Image Saved!")   

        # Replace this with the path to your image
        image = Image.open(user_input_filename).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = best_model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
        os.remove(user_input_filename)
        print("Image Removed!")
        return Detection_Result, prediction

    submit = st.button(label="Submit Sketch")
    if submit:
        st.subheader("Output")
        classified_label, prediction = predict_parkinsons(canvas_image)
        with st.spinner(text="This may take a moment..."):
            st.write(classified_label)

            class_names = open("labels.txt", "r").readlines()

            data = {
                "Class": class_names,
                "Confidence Score": prediction[0],
            }

            df = pd.DataFrame(data)

            df["Confidence Score"] = df["Confidence Score"].apply(
                lambda x: f"{str(np.round(x*100))[:-2]}%"
            )

            df["Class"] = df["Class"].apply(lambda x: x.split(" ")[1])

            st.subheader("Confidence Scores on other classes:")
            st.write(df)


if (selected == 'Visual Input Spiral Model'): 

    # Add the second part of the script
    st.header("Detecting Parkinson's Disease - Visual Input Spiral Model")

    st.write("Upload an image to classify into Healthy or Parkinson's.")
    st.warning("Warning: Supported image formats: PNG, JPG, JPEG.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Process the image and make a prediction
        if st.button("Classify"):
            # Save the uploaded image temporarily
            user_input_filename = "user_input.png"
            with open(user_input_filename, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the trained model
            model = load_model("keras_Model.h5", compile=False)

            # Load the labels
            class_names = open("labels.txt", "r").readlines()

            # Create the array of the right shape to feed into the Keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Open the uploaded image
            image = Image.open(user_input_filename).convert("RGB")

            # Resize the image to be at least 224x224 and then crop from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # Convert the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Make a prediction
            prediction = model.predict(data)
            confidence_score = prediction[0][0]  # Assuming 0 is the index for Parkinson's class

            # Display the result
            st.subheader("Classification Result:")
            if confidence_score >= 0.5:
                st.write("The model has classified the image as Healthy.")
            else:
                st.write("The model has classified the image as Parkinson's.")

            # Remove the temporary image file
            os.remove(user_input_filename)
