# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# from PIL import Image
# import requests
# from huggingface_hub import hf_hub_download
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# # Hugging Face model URL

# # url = "https://huggingface.co/madboi/TB_Detection-Model/resolve/main/tb_classification_model.h5"
# # model_path = "tb_classification_model.h5"

# # @st.cache_resource
# # def load_model_from_huggingface():
# #     try:
# #         # Download the model from your Hugging Face repo
# #         model_path = hf_hub_download(
# #             repo_id="madboi/TB_Detection-Model",  
# #             filename="tb_classification_model.h5",  
# #         )

# #         # Load the model
# #         model = tf.keras.models.load_model(model_path)
# #         return model

# #     except Exception as e:
# #         st.error(f"Model loading failed: {e}")
# #         st.stop()

# # Upload file widget should be outside the cached function
# uploaded_model = st.file_uploader("📥 Upload the model (.h5 file) manually", type=["h5"])

# @st.cache_resource
# def load_model(uploaded_file=None):
#     try:
#         # Try loading from Hugging Face
#         model_path = hf_hub_download(
#             repo_id="madboi/TB_Detection-Model",
#             filename="tb_classification_model.h5",
#         )
#         model = tf.keras.models.load_model(model_path)
#         return model

#     except Exception as e:
#         st.warning("🚨 Could not fetch model from Hugging Face.")

#         # Try to use uploaded model if available
#         if uploaded_file is not None:
#             with open("uploaded_model.h5", "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             model = tf.keras.models.load_model("uploaded_model.h5")
#             return model
#         else:
#             st.error("❌ No model available. Please upload the `.h5` model file.")
#             st.stop()


# # Load it once (cached)
# # model = load_model_from_huggingface()
# model = load_model(uploaded_model)



# def preprocess_image(img):
#     img = img.resize((224, 224))  # Resize to model input size
#     img_array = image.img_to_array(img)  

#     # Convert grayscale images to 3-channel RGB
#     if img_array.shape[-1] == 1:
#         img_array = np.repeat(img_array, 3, axis=-1)  # (224, 224, 1) → (224, 224, 3)

#     img_array = img_array / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     return img_array  # ✅ Return processed image


# # Function to compute Grad-CAM
# def compute_gradcam(model, img_array, layer_name="conv4_block5_out"):
#     grad_model = tf.keras.models.Model(inputs=model.input, 
#                                        outputs=[model.get_layer(layer_name).output, model.output])

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         class_idx = tf.argmax(predictions[0])
#         loss = predictions[:, class_idx]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs.numpy()[0]
#     heatmap = np.dot(conv_outputs, pooled_grads.numpy())

#     heatmap = np.maximum(heatmap, 0)  # ReLU
#     heatmap /= np.max(heatmap) + 1e-10  # Normalize
#     return heatmap

# # Function to overlay Grad-CAM heatmap
# def overlay_gradcam(img, heatmap, alpha=0.4):
#     """Overlay Grad-CAM heatmap on the original image."""
    
#     # Ensure heatmap is the same size as the image
#     heatmap = cv2.resize(heatmap, (img.width, img.height))  # Resize to match original image
    
#     # Normalize heatmap to [0,255] and apply color mapping
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Convert PIL image to NumPy array
#     original = np.array(img)

#     # If original image is grayscale, convert to RGB
#     if len(original.shape) == 2 or original.shape[-1] == 1:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

#     # Ensure both images have the same shape (H, W, 3)
#     if heatmap.shape != original.shape:
#         heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

#     # Blend original image with heatmap
#     superimposed = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

#     return superimposed



# # Streamlit UI
# st.title("Tuberculosis Detection using ResNet50")
# st.write("Upload a Chest X-ray image to classify as **Normal** or **TB**.")

# uploaded_file = st.file_uploader("Choose a Chest X-ray Image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image_pil = Image.open(uploaded_file)
#     st.image(image_pil, caption="Uploaded Image", use_container_width=True)

#     # Preprocess image
#     img_array = preprocess_image(image_pil)

#     # Get model prediction
#     prediction = model.predict(img_array)[0][0]
#     result = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
#     confidence = prediction if prediction > 0.5 else 1 - prediction

#     st.subheader(f"Prediction: **{result}**")
#     st.write(f"Confidence: **{confidence:.2%}**")

#     # Generate Grad-CAM
#     heatmap = compute_gradcam(model, img_array)
#     gradcam_output = overlay_gradcam(image_pil, heatmap)

#     # Show Grad-CAM
#     st.subheader("Grad-CAM Visualization")
#     st.image(gradcam_output, caption="Grad-CAM Heatmap", use_container_width=True)


import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.title("Tuberculosis Detection using ResNet50 (TFLite)")
st.write("📤 Upload a Chest X-ray image to classify as **Normal** or **TB**.")
st.write("📦 Also upload your **`.tflite` model file** below (only once):")

# Upload the .tflite model manually
uploaded_model = st.file_uploader("Upload `.tflite` model", type=["tflite"])



@st.cache_resource
def load_tflite_model_from_huggingface():
    try:
        model_path = hf_hub_download(
            repo_id="madboi/TB_Detection-Model",
            filename="tb_model_float16.tflite"
        )
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"🚨 Could not load model from Hugging Face: {e}")
        st.stop()

interpreter = load_tflite_model_from_huggingface()
lite_model(uploaded_model)

    # Preprocessing
    def preprocess_image(img):
        img = img.resize((224, 224))
        img_array = np.array(img)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        img_array = img_array.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    # Prediction
    def predict_tflite(interpreter, img_array):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Upload image
    uploaded_image = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_pil = Image.open(uploaded_image)
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(image_pil)
        prediction = predict_tflite(interpreter, img_array)

        result = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader(f"Prediction: **{result}**")
        st.write(f"Confidence: **{confidence:.2%}**")

  
   
