import random
from time import time

import cv2
from PIL import ImageFont


class ImageProcessor(object):
    def __init__(self, font_name, font_size=20):
        self.text_font = ImageFont.truetype(font_name, font_size)
        self.cv2_font = cv2.FONT_HERSHEY_SIMPLEX

        print('Loading Haar Cascades..')
        self.face_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_smile.xml')

        print('Loading neural network models...')
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['AGE_0-2', 'AGE_4-6', 'AGE_8-12', 'AGE_15-20', 'AGE_25-32', 'AGE_38-43', 'AGE_48-53', 'AGE_60-100']
        self.gender_list = ['MALE', 'FEMALE']

        self.age_net = cv2.dnn.readNetFromCaffe(
            "age_gender_models/deploy_age.prototxt",
            "age_gender_models/age_net.caffemodel")
        self.gender_net = cv2.dnn.readNetFromCaffe(
            "age_gender_models/deploy_gender.prototxt",
            "age_gender_models/gender_net.caffemodel")

        self.last_phrase_time = time()
        random.seed(time())

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

        detected_faces = list()

        # Detecting age, gender and smile
        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            face_img = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]

            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]

            # Detect smile
            smiles = bool(len(self.smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(15, 10))))

            detected_faces.append({
                'xy': (x, y, w, h),
                'age': age,
                'gender': gender,
                'smile': smiles
            })

        return detected_faces
