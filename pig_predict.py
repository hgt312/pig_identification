import os

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image

from utils import as_num


np.set_printoptions(precision=10, suppress=True)
data_path = '/media/hgt/share2/jdd/Pig_Identification_Qualification_Test_A/test_A'
model = load_model('./models/pig_1128_1.hdf5')
pig2class = {'pig11': 2, 'pig22': 14, 'pig16': 7, 'pig23': 15, 'pig19': 10,
             'pig21': 13, 'pig1': 0, 'pig7': 27, 'pig17': 8, 'pig14': 5,
             'pig20': 12, 'pig3': 22, 'pig27': 19, 'pig26': 18, 'pig25': 17,
             'pig12': 3, 'pig18': 9, 'pig10': 1, 'pig28': 20, 'pig4': 24,
             'pig30': 23, 'pig13': 4, 'pig24': 16, 'pig8': 28, 'pig2': 11,
             'pig9': 29, 'pig29': 21, 'pig6': 26, 'pig5': 25, 'pig15': 6}


def transfer(preds):
    result = np.zeros((30,))
    for i in range(1, 31):
        result[i-1] = preds[pig2class['pig' + str(i)]]
    return result


def predict(pig_model, img):
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # print(x)
    preds = pig_model.predict(x)
    preds = np.squeeze(preds)
    preds /= preds.sum() * 1.000001
    return transfer(preds)


file_list = list(os.path.join(data_path, name) for name in os.listdir(data_path))
with open('./results/loss_1128_1.csv', mode='w', encoding='utf-8') as f:
    f.truncate()
    for n, file in enumerate(file_list):
        file_name = file.split('/')[-1][:-4]
        img = Image.open(file)
        preds = predict(model, img)
        for i in range(1, 31):
            f.write(','.join([file_name, str(i), str(as_num(preds[i-1]))]) + '\r\n')
        if n % 100 == 0:
            print('done', n)

# zh = predict(model, Image.open('../train/pig2/1.jpg'))
# print(zh.argmax() + 1)
