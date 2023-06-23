import streamlit as st
import tensorflow as tf
import os

# streamlit run streamlit_app.py

#### SideBar ####

st.sidebar.title('What is "What Bird is That?"')
st.sidebar.write('''
"What Bird is That?" is a **CNN Image Classification Model** which identifies the type of bird in an image. 

It can accurately identify over 500 different types of birds!

It was developed by retraining and fine-tuning a pre-trained Image Classification Model on close to 100,000 photos of birds.

**Accuracy :** **`ENTER FINAL ACCURACY HERE`**

**Model :** **`EfficientNetB4`**

''') #  **Dataset :** **`Food101`**

st.sidebar.markdown("Created by **Noah Ripstein**")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://www.linkedin.com/in/noah-ripstein/" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="linkedin_logo" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/nripstein" target="blank"><img align="center" src="https://logos-world.net/wp-content/uploads/2020/11/GitHub-Emblem.png" alt="github_logo" height="40" width="64" /></a></th>


""", unsafe_allow_html=True)

#### Main Body ####

# use st.cache_resource to cache model! # https://docs.streamlit.io/library/advanced-features/caching

st.title("What Bird is That? 🦜 📸")
st.header("Identify what kind of bird you snapped a photo of!")
st.write("To learn more about this app and how it was designed, visit [**GitHub**](https://github.com/nripstein)")
file = st.file_uploader(label="Upload an image of a bird.",
                        type=["jpg", "jpeg", "png"])


