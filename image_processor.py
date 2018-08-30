import random
from time import time, sleep

import cv2
from PIL import Image, ImageFont, ImageDraw

FACE_SIZE_TO_OFFER_DRINK = 200

PHRASES = {
    'COME_CLOSER': [
        'Hey, I see you! Come closer!',
        "Don't be shy! Come closer and I'll choose a drink for you.",
        'Hey! Come here!'
    ],
    'AGE_0-2': [
        'Are you sure you are old enough to get a drink?'
    ],
    'AGE_4-6': [
        'Are you sure you are old enough to get a drink?'
    ],
    'AGE_8-12': [
        "Are you sure you are old enough to get a drink?"
    ],
    'AGE_15-20': [
        "You look so good that I'll have to ask you for an ID"
    ],
    'AGE_25-32': [
        "Looking for something to drink?"
    ],
    'AGE_38-43': [
        "Lookin' for a drink to relax after a busy day?"
    ],
    'AGE_48-53': [
        "Lookin' for something that helps you to relax after a busy day?"
    ],
    'AGE_60-100': [
        'Hello sir! Do you still get asked for ID?'
    ],
    'MALE': [
        'man',
        'dude',
        'fella',
        'buddy',
        'bro',
        'kiddo',
        'mate'
    ],
    'FEMALE': [
        'beautiful',
        'gorgeous',
        'female human',
        'madame'
    ]
}



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


    def process_frame(self, frame, width, height):
        detected_faces = self.detect_faces(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        text = ""

        if frame.height < height:
            frame = frame.resize((width - 2, height - 2))

        if not len(detected_faces):
            return frame, text

        face = detected_faces[0]
        random.seed(time())
        if (face['w'] >= FACE_SIZE_TO_OFFER_DRINK and face['h'] >= FACE_SIZE_TO_OFFER_DRINK) and time() - self.last_phrase_time > 10:
            text = f"Hello, {random.choice(PHRASES[face['gender']])}! {random.choice(PHRASES[face['age']])}"

            if face['gender'] == "FEMALE":
                text += " Would you like some wine?"
            else:
                text += " Would you like some beer?"

            frame.paste((0, 0, 0), [0, 0, frame.size[0], frame.size[1]])
            image_text = ImageDraw.Draw(frame)
            image_text.multiline_text((10, 10), text, font=self.text_font, fill=(0, 255, 0, 128))

            # We don't want Bartender to start calling people right after serving the client.
            self.last_phrase_time = time()
        else:
            if time() - self.last_phrase_time < 10:
                image_text = ImageDraw.Draw(frame)
                image_text.multiline_text((10, 10), "Please come closer...", font=self.text_font, fill=(0, 255, 0, 128))
            else:
                text = random.choice(PHRASES['COME_CLOSER'])
                frame.paste((0, 0, 0), [0, 0, frame.size[0], frame.size[1]])
                image_text = ImageDraw.Draw(frame)
                image_text.multiline_text((10, 10), text, font=self.text_font, fill=(0, 255, 0, 128))
                self.last_phrase_time = time()

        return frame, text

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

        detected_faces = list()

        # Draw a rectangle around every face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
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
            smiles = bool(len(self.smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(15, 15))))
            overlay_text = "%s, %s, smiles: %s" % (gender, age, smiles)
            cv2.putText(frame, overlay_text, (x-100, y), self.cv2_font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            detected_faces.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'age': age,
                'gender': gender,
                'smile': smiles
            })

        return detected_faces
