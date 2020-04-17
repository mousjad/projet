import numpy as np
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import imageio
import os, os.path
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


def resize_images(image_arrays, size=[32, 32]):
    # convert float type to integer 
    image_arrays = (image_arrays * 255).astype('uint8')

    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)


def main():
    # path = "anime_data/cropped"
    # valid_images = [".jpg", ".gif", ".png", ".tga"]
    # ls = os.listdir(path)
    # imgs = np.zeros((len(ls),64, 48, 3))
    # for f in range(len(ls)):
    #     ext = os.path.splitext(ls[f])[1]
    #     if ext.lower() not in valid_images:
    #         continue
    #
    #     img = Image.open(os.path.join(path, ls[f])).resize((48,64))
    #     imgs[f, :, :, :] = np.asarray(img)
    #
    # train_imgs, test_imgs = train_test_split(imgs, test_size=0.33, random_state=42)
    # #
    # train = {'X': train_imgs}
    # test = {'X': test_imgs}
    # #
    # save_pickle(train, 'anime_data/train.pkl')
    # save_pickle(test, 'anime_data/test.pkl')

    imgs = fetch_lfw_people(min_faces_per_person=3, color=True, data_home='face_data', resize=0.55).images[:, 0:64, 0:48, :]
    labels = fetch_lfw_people(min_faces_per_person=3, color=True, data_home='face_data').target
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=42)

    train = {'X': train_imgs,
             'Y': train_labels}
    test = {'X': test_imgs,
            'Y': test_labels}

    save_pickle(train, 'face_data/train.pkl')
    save_pickle(test, 'face_data/test.pkl')


if __name__ == "__main__":
    main()
