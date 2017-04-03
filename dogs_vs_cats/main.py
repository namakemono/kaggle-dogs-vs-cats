"""
References
----------
..[1]. Kaggle - Dogs vs. Cats Redux: Kernels Edition, <https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition>
..[2]. Kaggle - Tutorial: How to get 0.06 Loss in <70 lines of code, <https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/discussion/27950>
..[3]. Kaggle - Cats vs Dogs .05 pytorch example, <https://www.kaggle.com/nothxplz/dogs-vs-cats-redux-kernels-edition/cats-vs-dogs-05-pytorch-example/run/761413>
..[4]. Github - raghakot / keras - resnet, <https://github.com/raghakot/keras-resnet>
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import cv2
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import KFold

NUM_KFOLD = 10
NUM_USE_MODEL = 1
IMG_SIZE = 300
ROOT = os.path.dirname(os.path.abspath(__file__)) 
TRAIN_PATH = os.path.join(ROOT, "../temp/train")
VALID_PATH = os.path.join(ROOT, "../temp/validation")

def preprocess_input(x):
    return (x - np.array([103.939, 116.779, 123.68]))[...,::-1]

def makedirs_if_needed(target):
    if not os.path.exists(target):
        os.makedirs(target)

def split(symlink_to_dir, fpath_list):
    for fpath in fpath_list:
        fname = os.path.basename(fpath)
        label = "dog" if fname.startswith("dog") else "cat"
        os.symlink(fpath, os.path.join(symlink_to_dir, label, fname))

def rearrange(kfold_index):
    for target in ["train/dog", "train/cat", "validation/dog", "validation/cat"]:
        symlink_to_dir = os.path.join(ROOT, "../temp", target)
        makedirs_if_needed(symlink_to_dir)
        for fpath in glob.glob(os.path.join(symlink_to_dir, "*")):
            os.unlink(fpath)
    fpath_list = np.asarray(glob.glob(os.path.join(ROOT, "../input/train/*.jpg")))
    kf = KFold(n_splits=NUM_KFOLD, shuffle=True, random_state=7777)
    train_index, test_index = list(kf.split(fpath_list))[kfold_index]
    train_fpath_list, test_fpath_list = fpath_list[train_index], fpath_list[test_index]
    split(TRAIN_PATH, train_fpath_list)
    split(VALID_PATH, test_fpath_list)

def DogsVSCatsNet():
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    tail_input = Input(shape=input_shape)
    tail = Lambda(preprocess_input)(tail_input)
    body_model = ResNet50(include_top=False, input_shape=input_shape)
    for layer in body_model.layers:
        layer.trainable = False
    body = body_model(tail)
    body = Flatten()(body)
    body = Dense(2, activation="softmax")(body)
    model = Model(input=tail_input, output=body)
    return model 

def get_model_weights(kfold_index):
    return os.path.join(ROOT, "../bin/dogs_vs_cats_net[%d].h5" % kfold_index)

def train(kfold_index):
    rearrange(kfold_index)
    model = DogsVSCatsNet()
    model.load_weights(get_model_weights(kfold_index))
    datagen = ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.05,
            fill_mode="constant",
            channel_shift_range=10,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05
        )
    train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(IMG_SIZE, IMG_SIZE), batch_size=8, shuffle=True, follow_links=True)
    validation_generator = datagen.flow_from_directory(VALID_PATH, target_size=(IMG_SIZE, IMG_SIZE), batch_size=8, shuffle=True, follow_links=True)
    model.compile(Nadam(lr=1e-4), "categorical_crossentropy", metrics=["accuracy"])
    lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-7)
    checkpointer = ModelCheckpoint(get_model_weights(kfold_index), verbose=1, save_best_only=True, save_weights_only=True)
    callbacks = [lr_reducer, checkpointer]
    model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.nb_sample / 10,
            nb_epoch=30,
            validation_data=validation_generator,
            nb_val_samples=validation_generator.nb_sample,
            callbacks=callbacks)

def predict(kfold_index):
    X, ids = [], []
    model = DogsVSCatsNet()
    model.load_weights(get_model_weights(kfold_index))
    y_pred = None
    for fpath in glob.glob(os.path.join(ROOT, "../input/test1/*.jpg")):
        img = cv2.imread(fpath)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        id = int(os.path.splitext(os.path.basename(fpath))[0])
        X.append(img)
        ids.append(id)
        if len(X) == 100:
            y = model.predict(np.asarray(X, dtype=np.float32))[:, 1]
            y_pred = y if (y_pred is None) else np.r_[y_pred, y]
            X = []
    if len(X) > 0:
        y_pred = np.r_[y_pred, model.predict(np.asarray(X, dtype=np.float32))[:, 1]]
    return y_pred, ids

def make_submissions():
    y_pred_list, ids = [], []
    for kfold_index in range(NUM_USE_MODEL):
        y_pred, ids = predict(kfold_index)
        y_pred_list.append(y_pred)
    y_pred = np.mean(y_pred_list, axis=0)
    y_pred = y_pred.clip(0.001, 0.999)
    df = pd.DataFrame(zip(ids, y_pred), columns=["id", "label"])
    df.sort_values(by="id", inplace=True)
    df.to_csv(os.path.join(ROOT, "../output/submission.csv"), index=False)
    
def run():
    for kfold_index in range(NUM_USE_MODEL):
        train(kfold_index)
    make_submissions()

if __name__ == "__main__":
    run()

