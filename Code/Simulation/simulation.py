import csv
import sys

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import cv2
from pickle import load, dump
print(sys.version)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CSV_PATH = "C:\\Users\\hadas\\Desktop\\project\\simulation\\NSL-KDD\\NSL_KDD_sample.csv"
SCALAR_PATH = "C:\\Users\\hadas\\Desktop\\project\\simulation\\NSL-KDD\\scaler.pkl"
DATA = 'NSL-KDD'
TARGET_SIZE = (224, 224)
INPUT_SIZE = (224, 224, 3)
BATCHSIZE = 1  # could try 128 or 32


class NSL_KDD():

    def __init__(self, dic):
        self.df = pd.DataFrame.from_dict(dic)
        self.numeric_features = ['Src Bytes', 'Dst Bytes', 'Urgent', 'Num Failed Logins',
                                 'Root Shell', 'Num Shells', 'Is Hot Logins', 'Is Guest Login',
                                 'Diff Srv Rate', 'Srv Diff Host Rate',
                                 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate',
                                 'Difficulty Level']
        self.label_dic = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
        self.label = self.label_dic[self.df['Class'].tolist()[0]]

    def transform_features_kdd(self):
        self.df.drop(['Class'], axis=1, inplace=True)
        self.df = self.df[list(self.df.columns.values)].astype(float)
        scaler = load(open(SCALAR_PATH, 'rb'))
        self.df[self.numeric_features] = scaler.transform(self.df[self.numeric_features])
        # Multiply the feature values by 255 to transform them into the scale of [0,255]
        numeric_features1 = self.df.dtypes[self.df.dtypes != 'object'].index
        self.df[numeric_features1] = self.df[numeric_features1].apply(lambda x: (x * 255))


def make_img_from_df(df):
    ims = df.iloc[0].values
    ims = np.array(ims).reshape(5, 5, 1)
    array = np.array(ims, dtype=np.uint8)
    array = np.squeeze(array, axis=2)
    new_image = Image.fromarray(array)
    opencvImage = np.array(new_image)
    img = cv2.resize(opencvImage, (224, 224))
    # cv2.imshow('image window', img)
    # # add wait key. window waits until user presses a key
    # cv2.waitKey(0)
    # # and finally destroy/close all open windows
    # cv2.destroyAllWindows()
    return img


def make_image(dic, data):
    if data == 'NSL-KDD':
        data_NSL_KDD = NSL_KDD(dic)
        data_NSL_KDD.transform_features_kdd()
        img = make_img_from_df(data_NSL_KDD.df)
    return img, data_NSL_KDD.label


def image_generator(img):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_dataframe(
        pd.DataFrame(img),
        target_size=TARGET_SIZE,
        # color_mode=COLOR_MODE,
        batch_size=BATCHSIZE,
        class_mode='categorical')
    return generator


if __name__ == '__main__':
    with open(CSV_PATH) as file_obj:
        reader_obj = csv.reader(file_obj)
        for i, line in enumerate(reader_obj):
            if i == 0:
                column = line
            else:
                dic = {column[i]: [line[i]] for i in range(len(line))}
                # df = pd.DataFrame.from_dict(dic)
                # df.drop(['Class'],axis = 1,inplace=True)
                # df=df[list(df.columns.values)].astype(float)
                # print(df.info())
                img, label = make_image(dic, DATA)
                print(type(img))
                generator = image_generator(img)
                img = generator.next()
                # print(f'label{label.shape}')
                # print(label)
                print(img.shape)  # (1,256,256,3)
                # plt.imshow(img[0])
                # plt.show()
                break
