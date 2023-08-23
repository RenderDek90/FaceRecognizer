import os
import cv2 as cv
import numpy as np


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


def get_class_id(root_path, train_names):

    image_classes_list = []
    train_image_list = []

    for id, folder_name in enumerate(train_names):
        for img_names in os.listdir(root_path + '/' + folder_name):
            img = cv.imread(root_path + '/' + folder_name + '/' + img_names)
            train_image_list.append(img)
            image_classes_list.append(id)

    return train_image_list, image_classes_list

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

    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id

        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''


def train(train_face_grays, image_classes_list):

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(
        train_face_grays, np.array(image_classes_list))
    return recognizer

    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id

        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''


def get_test_images_data(test_root_path):
    test_image_list = []

    for img_name in os.listdir(test_root_path):
        img = cv.imread(test_root_path + '/' + img_name)
        test_image_list.append(img)

    return test_image_list

    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory

        Returns
        -------
        list
            List containing all loaded gray test images
    '''


def predict(recognizer, test_faces_gray):

    predict_results = []
    for img in test_faces_gray:
        result, _ = recognizer.predict(img)
        predict_results.append(result)

    return predict_results
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''


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
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''


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


'''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''


'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == '__main__':

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = 'dataset/train'
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(
        train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(
        train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = 'dataset/test'
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(
        test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(
        predict_results, test_image_list, test_faces_rects, train_names)

    combine_and_show_result(predicted_test_image_list)
