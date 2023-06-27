# What Bird is That? 🦜 📸
Computer vision website which identifies the species of birds in user-uploaded photos
# [Try the app (currently in beta testing)](https://what-bird-is-that.streamlit.app/)

# Sample Usage




https://github.com/nripstein/What-Bird-is-That/assets/98430636/d81f35ee-5861-441e-92ea-70b1de88bbda






# This project is in early stages, and changes are still activley being made

# todo (for initial "release"):
- [X] Deploy streamlit app
- [X] Make links to the wikipedia page of the top 3 predicted classes
- [ ] Finish training model
- [ ] Add model training notebook to repo
- [ ] Add video demonstration to repo
- [ ] Add accuracy metric to app
- [ ] Add github logo with link to my github to app

# future plans
- [ ] make script for scraping other types of birds so it can classify more (no white swan or flamingo included in dataset)
- [ ] Ideally scrape habitat location images from wikipedia (seems very hard after a few hours of trying because there's no consistent naming convention)


# version 2:
1. introduce autocrop option using YOLOv5 object detection

Add options for secondary models (like b0-b4)

# ideas for better computer vision models:
1. try data augmentation using tf.ImageDataGenerator instead of a sequential augmentation layer built into the model itself
2. get equal number of images for each class using data augmentation

