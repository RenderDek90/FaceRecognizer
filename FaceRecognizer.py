import os
import cv2 as cv
import numpy as np


def get_path_list(root_path):

    list_name_folder = []

    for name in os.listdir(root_path):
        list_name_folder.append(name)

    return list_name_folder


def get_class_id(root_path, train_names):

    image_classes_list = []
    train_image_list = []

    for id, folder_name in enumerate(train_names):
        for img_names in os.listdir(root_path + '/' + folder_name):
            img = cv.imread(root_path + '/' + folder_name + '/' + img_names)
            train_image_list.append(img)
            image_classes_list.append(id)

    return train_image_list, image_classes_list


def detect_faces_and_filter(image_list, image_classes_list=None):

    face_cascade = cv.CascadeClassifier(
        './haarcascades/haarcascades/haarcascade_frontalface_default.xml')

    train_face_grays = []
    train_faces_rects = []
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
            if (image_classes_list != None):
                filtered_classes_list.append(image_classes_list[id])
            train_faces_rects.append(face_rects)

    return train_face_grays, train_faces_rects, filtered_classes_list


def train(train_face_grays, image_classes_list):

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(
        train_face_grays, np.array(image_classes_list))
    return recognizer


def get_test_images_data(test_root_path):
    test_image_list = []

    for img_name in os.listdir(test_root_path):
        img = cv.imread(test_root_path + '/' + img_name)
        test_image_list.append(img)

    return test_image_list


def predict(recognizer, test_faces_gray):

    predict_results = []
    for img in test_faces_gray:
        result, _ = recognizer.predict(img)
        predict_results.append(result)

    return predict_results


def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):

    predicted_test_image_list = []
    for idx, img in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[idx]
        text = train_names[predict_results[idx]]
        if (train_names[predict_results[idx]] == 'Jokowi'):
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)
            cv.putText(img, text + " (Active)", (x-10, y+h+50),
                       cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), thickness=3, lineType=cv.LINE_AA)
        else:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 10)
            cv.putText(img, text + " (Inactive)", (x-10, y+h+50),
                       cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        predicted_test_image_list.append(img)

    return predicted_test_image_list


def combine_and_show_result(image_list):

    resized_image = []
    for id, name in enumerate(image_list):
        scale = 0.4
        newHeight = int(image_list[id].shape[0] * scale)
        newWidth = int(image_list[id].shape[1] * scale)
        newDimension = (newWidth, newHeight)
        img = cv.resize(image_list[id], newDimension, cv.INTER_AREA)
        resized_image.append(img)

    row_1 = np.hstack([resized_image[0], resized_image[1]])
    row_2 = np.hstack([resized_image[3], resized_image[4]])
    col_combine = np.vstack([row_1, row_2])
    result = np.hstack([col_combine, resized_image[2]])

    cv.imshow('Result', result)
    cv.waitKey(0)


if __name__ == '__main__':

    train_root_path = 'dataset/train'

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(
        train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(
        train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    test_root_path = 'dataset/test'

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(
        test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(
        predict_results, test_image_list, test_faces_rects, train_names)

    combine_and_show_result(predicted_test_image_list)
