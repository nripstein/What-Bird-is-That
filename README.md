# What Bird is That? 🦜 📸
Computer vision website which identifies the species of birds by user-uploaded photos

# This project is in very early stages

# todo (for initial "release"):
- [ ] Finish training model
- [ ] Add model training notebook to repo
- [X] Deploy streamlit app
- [ ] Add video demonstration to repo
- [ ] Add accuracy metric to app
- [ ] Add github logo with link to my github to app

# future plans
- [ ] make script for scraping other types of birds so it can classify more (no white swan or flamingo included in dataset)

# version 2:
1. introduce autocrop option using YOLOv5 object detection

Add options for secondary models (like b0-b4)

# ideas for better computer vision models:
1. try data augmentation using tf.ImageDataGenerator instead of a sequential augmentation layer built into the model itself
2. get equal number of images for each class using data augmentation
