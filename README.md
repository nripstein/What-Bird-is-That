
# What Bird is That? 🦜 📸
A computer vision website which identifies the species of birds in user-uploaded photos. [Try the website at this link](https://what-bird-is-that.streamlit.app/).

# Sample Usage

https://github.com/nripstein/What-Bird-is-That/assets/98430636/d81f35ee-5861-441e-92ea-70b1de88bbda

# Summary
This project aims to classify bird species using deep learning techniques. By leveraging transfer learning and fine-tuning, the Bird Classifier feature extraction and fine tuning of the [EfficientNetB4](https://arxiv.org/pdf/1905.11946.pdf) model, pre-trained on the [ImageNet dataset](https://www.image-net.org/), to accurately identify and provide information about various bird species.

# Deep learning model design

1.  **Data augmentation:** Data augmentation is a powerful technique employed to increase the diversity and variability of the training dataset. The augmentation techniques employed include random horizontal flipping, random height and width shifting, random zooming, random rotation, and random contrast adjustment. By randomly applying these operations to each image during training, the model becomes more resilient to variations in bird pose, lighting conditions, and other factors. This augmentation process enhances the model's ability to generalize well and accurately classify bird species under different circumstances.
2.  **Feature extraction with EfficientNet**: Transfer learning is employed to leverage the knowledge gained from the extensive training on the large-scale ImageNet dataset. The EfficientNet architecture, renowned for its excellent performance and low training time in image classification tasks, serves as the backbone of the model.  Feature extraction uses the backbone model architecture and weights, but adds a few layers which are trained on the bird image dataset.
    
3.  **Fine-tuning the model**: After training the new layers during feature extraction, the weights of the last 10 layers of the EfficientNet model are unfrozen and trained on the bird images for an additional 5 epochs (with a reduced learning rate). This keeps most of learned features within those layers the same, but slightly adjusts them to be better at classifying birds specifically.

# Performance on unseen test data:

<table>
  <tr>
    <td>Accuracy</td>
    <td>97.83%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>98.19%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>97.82%</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>97.79%</td>
  </tr>
</table>





# Streamlit Website

To provide a user-friendly interface for bird classification, I developed a Streamlit web application. Users can easily upload their bird photos and obtain predictions from the trained Bird Classifier model.  The website presents a bar plot of the probability of the top 3 species (with their labels serving as Wikipedia links). After identifying the top prediction the website presents a photo of the bird from the test data set, and provides details about the recognized species (from the Wikipedia API).

### Example of Post-Classification Description
<div align="center">
  <img src="https://github.com/nripstein/What-Bird-is-That/assets/98430636/eadecb26-e345-472f-87e3-975d8f7bae49" alt="classified bird demo" style="width: 75%;">
</div>


# Data
The bird [dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) utilized in this project comprises a wide range of bird species, offering a comprehensive coverage of avian biodiversity. It includes 525 different species, enabling the model to accurately identify and classify a diverse range of birds.

### Sample images from data set

<div align="center">

| ![scarlet macaw](https://github.com/nripstein/What-Bird-is-That/assets/98430636/1b852732-b90f-4003-8ad2-aa0a101bfcce) | ![bald eagle](https://github.com/nripstein/What-Bird-is-That/assets/98430636/7d78b96d-4819-416c-ae54-206e1773b930) | ![blue dacnis](https://github.com/nripstein/What-Bird-is-That/assets/98430636/2c1742a1-b135-4cf1-a0d3-270bdca57750) |
|:---:|:---:|:---:|
| Scarlet Macaw | Bald Eagle | Blue Dacnis |

</div>

<!---



# This project is an ongoing work in progress

# todo:
- [X] Deploy streamlit app
- [X] Make links to the wikipedia page of the top 3 predicted classes
- [ ] Finish training correct model
- [ ] Add model training notebook to repo
- [ ] Add video demonstration to repo
- [ ] Add accuracy metric to app
- [ ] Add github logo with link to my github to app

# future plans
- [ ] make script for scraping other types of birds so it can classify more (no white swan or flamingo included in dataset)
- [ ] Ideally scrape habitat location images from wikipedia (seems very hard after a few hours of trying because there's no consistent naming convention)
- [ ] if length of wikipedia summary section is too short, use the description [Wood Duck is good example of very short summary page and longer description](https://en.wikipedia.org/wiki/Wood_duck)


# version 2:
1. introduce autocrop option using YOLOv5 object detection

Add options for secondary models (like b0-b4)

# ideas for better computer vision models:
1. try data augmentation using tf.ImageDataGenerator instead of a sequential augmentation layer built into the model itself
2. get equal number of images for each class using data augmentation

--->
