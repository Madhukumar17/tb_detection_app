import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import requests
from huggingface_hub import hf_hub_download
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Hugging Face model URL

# url = "https://huggingface.co/madboi/TB_Detection-Model/resolve/main/tb_classification_model.h5"
# model_path = "tb_classification_model.h5"

@st.cache_resource
def load_model_from_huggingface():
    try:
        # Download the model from your Hugging Face repo
        model_path = hf_hub_download(
            repo_id="madboi/TB_Detection-Model",  
            filename="tb_classification_model.h5",  
        )

        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# Load it once (cached)
model = load_model_from_huggingface()

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)  

    # Convert grayscale images to 3-channel RGB
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)  # (224, 224, 1) → (224, 224, 3)

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array  # ✅ Return processed image


# Function to compute Grad-CAM
def compute_gradcam(model, img_array, layer_name="conv4_block5_out"):
    grad_model = tf.keras.models.Model(inputs=model.input, 
                                       outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    heatmap = np.dot(conv_outputs, pooled_grads.numpy())

    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) + 1e-10  # Normalize
    return heatmap

# Function to overlay Grad-CAM heatmap
def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    
    # Ensure heatmap is the same size as the image
    heatmap = cv2.resize(heatmap, (img.width, img.height))  # Resize to match original image
    
    # Normalize heatmap to [0,255] and apply color mapping
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert PIL image to NumPy array
    original = np.array(img)

    # If original image is grayscale, convert to RGB
    if len(original.shape) == 2 or original.shape[-1] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    # Ensure both images have the same shape (H, W, 3)
    if heatmap.shape != original.shape:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    # Blend original image with heatmap
    superimposed = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

    return superimposed



# Streamlit UI
st.title("Tuberculosis Detection using ResNet50")
st.write("Upload a Chest X-ray image to classify as **Normal** or **TB**.")

uploaded_file = st.file_uploader("Choose a Chest X-ray Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = preprocess_image(image_pil)

    # Get model prediction
    prediction = model.predict(img_array)[0][0]
    result = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: **{result}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Generate Grad-CAM
    heatmap = compute_gradcam(model, img_array)
    gradcam_output = overlay_gradcam(image_pil, heatmap)

    # Show Grad-CAM
    st.subheader("Grad-CAM Visualization")
    st.image(gradcam_output, caption="Grad-CAM Heatmap", use_container_width=True)

# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# import os
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# from huggingface_hub import hf_hub_download
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# # Disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# MODEL_FILENAME = "tb_classification_model.h5"

# # Load model from Hugging Face or fallback to local
# @st.cache_resource
# def load_model():
#     try:
#         model_path = hf_hub_download(
#             repo_id="madboi/TB_Detection-Model",
#             filename=MODEL_FILENAME,
#             cache_dir="."
#         )
#     except Exception as e:
#         st.warning(f"Could not fetch from Hugging Face. Trying local fallback… ({e})")
#         if not os.path.exists(MODEL_FILENAME):
#             st.error("Model file not found locally. Please upload it or connect to the internet.")
#             st.stop()
#         model_path = MODEL_FILENAME

#     return tf.keras.models.load_model(model_path)

# model = load_model()

# # Preprocessing function
# def preprocess_image(img):
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     if img_array.shape[-1] == 1:
#         img_array = np.repeat(img_array, 3, axis=-1)
#     img_array = img_array / 255.0
#     return np.expand_dims(img_array, axis=0)

# # Grad-CAM computation
# def compute_gradcam(model, img_array, layer_name="conv4_block5_out"):
#     grad_model = tf.keras.models.Model(
#         inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         class_idx = tf.argmax(predictions[0])
#         loss = predictions[:, class_idx]
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs.numpy()[0]
#     heatmap = np.dot(conv_outputs, pooled_grads.numpy())
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap) + 1e-10
#     return heatmap

# # Grad-CAM overlay
# def overlay_gradcam(img, heatmap, alpha=0.4):
#     heatmap = cv2.resize(heatmap, (img.width, img.height))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     original = np.array(img)
#     if len(original.shape) == 2 or original.shape[-1] == 1:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
#     heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
#     superimposed = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
#     return superimposed

# # LIME prediction wrapper
# def predict_fn(images):
#     return model.predict(np.array(images))

# # Streamlit UI
# st.set_page_config(page_title="TB Detection", layout="centered")
# st.title("Tuberculosis Detection using ResNet50")
# st.write("Upload a Chest X-ray image to classify as **Normal** or **Tuberculosis (TB)**.")

# uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image_pil = Image.open(uploaded_file).convert("RGB")
#     st.image(image_pil, caption="Uploaded Image", use_container_width=True)

#     # Preprocessing
#     img_array = preprocess_image(image_pil)

#     # Prediction
#     prediction = model.predict(img_array)[0][0]
#     result = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
#     confidence = prediction if prediction > 0.5 else 1 - prediction
#     st.subheader(f"Prediction: **{result}**")
#     st.write(f"Confidence: **{confidence:.2%}**")

#     # Grad-CAM
#     heatmap = compute_gradcam(model, img_array)
#     gradcam_image = overlay_gradcam(image_pil, heatmap)

#     # LIME
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         np.squeeze(img_array[0]),
#         classifier_fn=predict_fn,
#         top_labels=2,
#         hide_color=0,
#         num_samples=1000
#     )
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=True,
#         num_features=5,
#         hide_rest=False
#     )
#     lime_image_output = mark_boundaries(temp / 255.0, mask)

#     # Display both
#     st.subheader("Model Interpretability")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
#     with col2:
#         st.image(lime_image_output, caption="LIME Explanation", use_container_width=True)


  
   
