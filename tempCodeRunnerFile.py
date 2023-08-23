import os
import cv2 as cv
import numpy as np


root_path = 'dataset/train'


def get_path_list(root_path):

    list_name_folder = []

    for name in os.listdir(root_path):
        list_name_folder.append(name)

    return list_name_folder

    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory

        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''


train_names = get_path_list(root_path)
# print(get_path_list(root_path='dataset/train'))


def get_class_id(root_path, train_names):

    list_id = []
    list_train_images = []

    for id, folder_name in enumerate(train_names):
        for img_names in os.listdir(root_path + '/' + folder_name):
            img = cv.imread(root_path + '/' + folder_name + '/' + img_names)
            list_train_images.append(img)
            list_id.append(id)

    return list_train_images, list_id

    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''


image_list, b = get_class_id(root_path, train_names)
# print(b)


def detect_faces_and_filter(image_list, image_classes_list=None):

    face_cascade = cv.CascadeClassifier(
        './haarcascades/haarcascades/haarcascade_frontalface_default.xml')

    train_face_grays = []
    test_faces_rects = []
    filtered_classes_list = []

    for id, img in enumerate(image_list):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detected_face = face_cascade.detectMultiScale(
            img_gray, scaleFactor=1.20, minNeighbors=5)

        if (len(detected_face) < 1):
            continue

        for face in detected_face:
            x, y, h, w = face
            face_rects = x, y, h, w
            face_img_gray = img_gray[y:y+h, x:x+w]

            train_face_grays.append(face_img_gray)
            filtered_classes_list.append(id)
            test_faces_rects.append(face_rects)

    return train_face_grays, test_faces_rects, filtered_classes_list


train_face_grays, test_faces_rects, filtered_classes_list = detect_faces_and_filter(
    image_list, image_classes_list=None)

print(test_faces_rects)