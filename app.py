import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import json
import wikipedia
from PIL import Image
import io
import base64


# streamlit run app.py  # run this for local running

st.set_page_config(page_title="What Bird is That?")  # Set tab title

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
def load_label_to_scientific(json_file_name: str = "common_to_scientific.json") -> dict[str, str]:
    """
    Loads a JSON file containing the label to scientific name mapping and returns it as a dictionary.

    Args:
        json_file_name (str): The path to the JSON file.

    Returns:
        dict: The label to scientific name mapping as a dictionary.
    """
    with open(f"models and data/{json_file_name}", "r") as json_file:
        label_to_scientific_dict = json.load(json_file)
    return label_to_scientific_dict


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "bird_model_b4_used_b2_imsize.h5") -> tf.keras.Model:
    tf_model = tf.keras.models.load_model(f"models and data/{model_name}")
    return tf_model


def display_bird_summary(best_guess_row: pd.Series) -> None:
    """
    Displays the bird summary, including the Wikipedia description and an image.

    Retrieves the Wikipedia description of the bird based on its scientific name,
    and displays it alongside an image of the bird.

    The image is wrapped in text, and is displayed on the right side.

    Args:
    best_guess_row (pd.Series): A pandas Series representing the best guess for the bird species.
                                It should contain the columns "Common Name" and "Scientific Name".
    """
    wiki_description = get_bird_description(best_guess_row["Scientific Name"])
    if wiki_description is not None:  # only is None if can't find species, so we shouldn't do anything
        other_image = Image.open(f"models and data/sample photos/{best_guess_row['Common Name']}.jpg")

        image_width = 300
        image_bytes = io.BytesIO()
        other_image.save(image_bytes, format="JPEG")
        image_html = f'<img src="data:image/jpeg;base64,{base64.b64encode(image_bytes.getvalue()).decode()}" ' \
                     f'alt="Bird Image" style="float: right; width: {image_width}px; margin-left: 20px;">'

        st.markdown(f'{image_html} {wiki_description}', unsafe_allow_html=True)


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
        img (bytes): The image to be classified.
        model (tf.keras.Model): The pre-trained model to use for prediction.

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
        "Common Name": labels,
        "Probability": values,
    })

    # Sort the DataFrame by Probability
    return prediction_df.sort_values("Probability", ascending=False)


@st.cache_data
def get_bird_description(scientific_name):
    try:
        return wikipedia.page(scientific_name).summary
    except wikipedia.exceptions.PageError:
        return None


def add_wikipedia(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df["Scientific Name"] = input_df["Common Name"].map(labels_to_sci)
    input_df["Wikipedia Link"] = input_df["Scientific Name"].apply(lambda species_name: 'https://en.wikipedia.org/wiki/' + species_name.lower().replace(' ', '_'))
    return input_df


labels_to_sci = load_label_to_scientific()
class_labels = sorted(list(labels_to_sci.keys()))
# ---------------------------------- SideBar ----------------------------------

st.sidebar.title('What is "What Bird is That?"')
st.sidebar.write('''
"What Bird is That?" is a **CNN Image Classification Model** which identifies the type of bird in an image. 

It can accurately identify over 500 different types of birds!

It was developed using close to 100,000 photos of birds using for transfer learning and fine-tuning of a pre-trained Image Classification Model.


''')
# **Accuracy :** **`ENTER FINAL ACCURACY HERE`**
#
# **Model :** **`EfficientNetB4`**

st.sidebar.markdown("Created by [**Noah Ripstein**](https://www.noahripstein.com)")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://www.linkedin.com/in/noah-ripstein/" target="blank">
<img align="center" src="https://bit.ly/3wCl82U" alt="linkedin_logo" height="40" width="40" /></a></th>

""", unsafe_allow_html=True)

# st.sidebar.image(open("app_images/tmp.png", "rb").read(), caption="GitHub Logo", width=64)

# ---------------------------------- Main Body ----------------------------------

st.title("What Bird is That? 🦜 📸")
st.header("Identify what kind of bird you snapped a photo of!")
st.write("To learn more about this website and the underlying machine learning model, "
         "[**read the writeup.**](https://www.noahripstein.com/project-1.html)")
file = st.file_uploader(label="Upload an image of a bird.",
                        type=["jpg", "jpeg", "png"])

if not file:
    image = None  # set to None because it doesn't exist yet
    pred_button = st.button("Identify Species", disabled=True, help="Upload an image to make a prediction")
    st.stop()
else:
    image = file.read()
    # Center the image using st.columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Display image in center column
        st.image(image, use_column_width="auto", width="50%")
    pred_button = st.button("Identify Species")

if pred_button:
    # Perform image classification and obtain prediction, confidence, and DataFrame
    with st.spinner("Loading Image Classification Model..."):
        model = load_model()

    with st.spinner("Classifying Image..."):
        df = classify_image(image, model)

    df = add_wikipedia(df)
    top_prediction_row = df.iloc[0]
    # Display the prediction and confidence
    st.success(f'Predicted Species: **{top_prediction_row["Common Name"].title()}** Confidence: {top_prediction_row["Probability"]:.2f}%')  # add something with scientific name here

    # create list for y-axis of plot which displays the name of the bird and links to the wikipedia page
    y_axis_wiki_namelist = []
    for index, row in df.iterrows():
        label = row["Common Name"]
        current_link = row["Wikipedia Link"]
        y_axis_wiki_namelist.append(f'<a href="{current_link}" target="_blank">{label}</a>')

    fig = go.Figure(data=[
        go.Bar(
            x=df["Probability"],
            y=y_axis_wiki_namelist,
            orientation="h",
            text=df["Probability"].apply(lambda x: f"{x:.2f}%"),
            textposition="auto",
            marker=dict(color="plum"),  # need to decide on a good colour
        )
    ])

    fig.update_layout(
        title="Common Name",
        xaxis_title="Probability",
        yaxis_title="Species",
        width=600,
        height=400,
        dragmode=False,
    )

    st.plotly_chart(fig)

    display_bird_summary(best_guess_row=top_prediction_row)


    # this is way too slow, so II'm going to have to accept the whole long thing, or pay for inference using GPT3 or something
    # from transformers import pipeline
    # @st.cache_resource(show_spinner=False)
    # def load_pipeline():
    #     return pipeline("summarization", model="facebook/bart-large-cnn")
    #
    #
    # @st.cache_data(show_spinner=False)
    # def summarize_with_pipeline(_summarizer, to_summarize: str) -> str:
    #     return _summarizer(to_summarize, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    # with st.spinner("Loading Text Summarization Model..."):
    #     summarization_model = load_pipeline()
    #
    # with st.spinner(f"Finding and Summarizing Some Information About {df.iloc[0]['Common Name'].title()}..."):
    #     summarized_description = summarize_with_pipeline(summarization_model, wiki_description)
    # st.markdown(summarized_description)
