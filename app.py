import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go


# streamlit run app.py  # run this for local running

# Hide hamburger menu and Streamlit watermark
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ---------------------------------- Functions ----------------------------------
@st.cache_data
def get_class_labels(labels_file: str = "class_labels.txt") -> list[str]:
    """Load the class labels from the text file"""
    labels = []
    with open(labels_file, "r") as txt_file:
        for line in txt_file:
            label = line.strip()  # Remove leading/trailing whitespaces
            labels.append(label)
    return labels


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "bird_model_b4_used_b2_imsize.h5") -> tf.keras.Model:
    tf_model = tf.keras.models.load_model(f"models/{model_name}")
    return tf_model


@st.cache_resource
def prep_image(img: bytes, shape: int = 260, scale: bool = False) -> tf.Tensor:
    """
    Preprocesses the image data.

    Args:
        img (bytes): The image data as a byte string.
        shape (int): The desired shape for the image (default: 260).
        scale (bool): Whether to scale the pixel values (default: False).

    Returns:
        The preprocessed image as a TensorFlow tensor.
    """
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=([shape, shape]))
    if scale:
        img = img/255.
    return img


def classify_image(img: bytes, model: tf.keras.Model) -> pd.DataFrame:
    """
    Classifies the given image using the provided model and returns a DataFrame
    containing the top 3 predictions and their probabilities.

    Args:
        img (tf.Tensor): The image to be classified.
        model: The pre-trained model to use for prediction.

    Returns:
        A pandas DataFrame containing the top 3 predictions and their probabilities,
        sorted in descending order of probability.
    """
    # Preprocess the image
    img = prep_image(img)
    # Expand dimensions to create a batch of size 1
    img = tf.cast(tf.expand_dims(img, axis=0), tf.int16)

    # Make predictions using the model
    pred_probs = model.predict(img)

    # Get the indices of the top 3 predicted classes
    top_3_indices = sorted(pred_probs.argsort()[0][-3:][::-1])

    # Compute the probabilities for the top 3 predictions
    values = pred_probs[0][top_3_indices] * 100
    labels = [class_labels[i] for i in top_3_indices]

    # Create a DataFrame to store the top 3 predictions and their probabilities
    prediction_df = pd.DataFrame({
        "Top 3 Predictions": labels,
        "Probability": values,
    })

    # Sort the DataFrame by Probability
    return prediction_df.sort_values("Probability", ascending=False)


class_labels = get_class_labels()
# ---------------------------------- SideBar ----------------------------------

st.sidebar.title('What is "What Bird is That?"')
st.sidebar.write('''
"What Bird is That?" is a **CNN Image Classification Model** which identifies the type of bird in an image. 

It can accurately identify over 500 different types of birds!

It was developed by retraining and fine-tuning a pre-trained Image Classification Model on close to 100,000 photos of birds.

**Accuracy :** **`ENTER FINAL ACCURACY HERE`**

**Model :** **`EfficientNetB4`**
''')

st.sidebar.markdown("Created by **Noah Ripstein**")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://www.linkedin.com/in/noah-ripstein/" target="blank">
<img align="center" src="https://bit.ly/3wCl82U" alt="linkedin_logo" height="40" width="40" /></a></th>

<th style="border:None"><a href="https://github.com/nripstein" target="blank"><img align="center" src="https://github.com/nripstein/What-Bird-is-That/blob/main/app_images/github_logo.png" alt="github_logo1tmp" height="40" width="64" /></a></th>

""", unsafe_allow_html=True)

# st.sidebar.image(open("app_images/tmp.png", "rb").read(), caption="GitHub Logo", width=64)

# ---------------------------------- Main Body ----------------------------------

st.title("What Bird is That? 🦜 📸")
st.header("Identify what kind of bird you snapped a photo of!")
st.write("To learn more about this website and the underlying machine learning model, "
         "[**visit the GitHub repository**](https://github.com/nripstein/What-Bird-is-That)")
file = st.file_uploader(label="Upload an image of a bird.",
                        type=["jpg", "jpeg", "png"])

if not file:
    image = None  # set to None because it doesn't exist yet
    pred_button = st.button("Predict", disabled=True, help="Upload an image to make a prediction")
    st.stop()

else:
    image = file.read()
    # Center the image using st.columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Apply CSS styling to center the image
        st.image(image, use_column_width="auto", width="50%")
    pred_button = st.button("Predict")


# Check if the prediction button is clicked
if pred_button:
    # Perform image classification and obtain prediction, confidence, and DataFrame
    with st.spinner("Loading Machine Learning Model..."):
        model = load_model()

    with st.spinner("Classifying Image..."):
        df = classify_image(image, model)

    # Display the prediction and confidence
    st.success(f'Prediction: {df.iloc[0]["Top 3 Predictions"]}\nConfidence: {df.iloc[0]["Probability"]:.2f}%')

    fig = go.Figure(data=[
        go.Bar(
            x=df["Probability"],
            y=df["Top 3 Predictions"],  # [f'<a href="https://en.wikipedia.org/wiki/Expected_value" target="_blank">{label}</a>' for label in df["Top 3 Predictions"]] # for making it clickable links
            orientation="h",
            text=df["Probability"].apply(lambda x: f"{x:.2f}%"),
            textposition="auto",
        )
    ])

    fig.update_layout(
        title="Top 3 Predictions",
        xaxis_title="Probability",
        yaxis_title="Species",
        width=600,
        height=400,
        dragmode=False,
    )

    st.plotly_chart(fig)
