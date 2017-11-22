import os

from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
from PIL import Image


data_path = '/media/hgt/share2/jdd/Pig_Identification_Qualification_Test_A/test_A'

model = load_model('/media/hgt/share2/jdd/try_time_1st.h5')


def predict(pig_model, img):
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    x = preprocess_input(x)
    preds = pig_model.predict(x)
    print(preds)


file_list = list(os.path.join(data_path, name) for name in os.listdir(data_path))
for file in file_list[:10]:
    file_name = file.split('/')[-1][:-4]
    img = Image.open(os.path.join(data_path, file_name))
    predict(model, img)
