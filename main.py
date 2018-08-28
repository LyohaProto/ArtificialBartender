import random
from time import sleep, time

import cv2
import pyttsx3

APP_WINDOW_TITLE = "Artificial Bartender"

# Configuring text-to-speech
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
rate = tts_engine.getProperty('rate')
tts_engine.setProperty('rate', rate - 75)

FACE_SIZE_TO_OFFER_DRINK = 250
WEBCAM_RES_W = 800
WEBCAM_RES_H = 600

age_list=['AGE_0-2','AGE_4-6','AGE_8-12','AGE_15-20','AGE_25-32','AGE_38-43','AGE_48-53','AGE_60-100']
gender_list = ['MALE', 'FEMALE']

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
        'Hello sir!'
    ],
    'MALE': [
        'man',
        'dude',
        'fella'
    ],
    'FEMALE': [
        'woman',
        'madame'
    ]
}


def init_webcam():
    # Initializing the webcam
    vs = cv2.VideoCapture(0)
    vs.set(3, WEBCAM_RES_W)  # CV_CAP_PROP_FRAME_WIDTH
    vs.set(4, WEBCAM_RES_H)  # CV_CAP_PROP_FRAME_HEIGHT

    ret = False
    retry = 0
    while not ret:
        ret, frame = vs.read()
        sleep(0.1)
        retry += 1
        assert retry < 30, 'Failed to initialize webcam'

    return vs


def say_random(text_list, min_interval):
    global last_phrase_time
    if globals().get('last_phrase_time') and time() - last_phrase_time < min_interval:
        return

    tts_engine.say(random.choice(text_list))
    tts_engine.runAndWait()
    last_phrase_time = time()


def capture_loop():
    print('Loading network models...')
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_net = cv2.dnn.readNetFromCaffe(
        "age_gender_models/deploy_age.prototxt",
        "age_gender_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "age_gender_models/deploy_gender.prototxt",
        "age_gender_models/gender_net.caffemodel")

    print('Loading Haar Cascades')
    faceCascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')
    smileCascade = cv2.CascadeClassifier('haar_cascades/haarcascade_smile.xml')

    print('Initializing webcam')
    # grab the reference to the webcam
    webcam = init_webcam()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # capture frames from the camera
    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

        # Draw a rectangle around every found face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_gray = gray[y:y + h, x:x + w]
            face_img = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            # Detect smile
            smiles = bool(len(smileCascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(15, 15))))
            overlay_text = "%s, %s, smiles: %s" % (gender, age, smiles)
            cv2.putText(frame, overlay_text, (x-100, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if len(faces):
            if (w >= FACE_SIZE_TO_OFFER_DRINK and h >= FACE_SIZE_TO_OFFER_DRINK):
                webcam.release()  # Stopping the webcam, otherwise it keeps capturing frames.

                # Displaying the captured face
                cv2.imshow(APP_WINDOW_TITLE, frame)
                cv2.waitKey(10)

                mood = "" if smiles else "Why so serious?"
                tts_engine.say(f'Hello, {random.choice(PHRASES[gender])}! {mood} {random.choice(PHRASES[age])}')
                if smiles:
                    tts_engine.say(f'Would you like some beer?')
                else:
                    tts_engine.say(f'Would you like some whiskey?')
                tts_engine.runAndWait()
                sleep(4)  # Waiting for the person to go away :)
                webcam = init_webcam()  # Starting the webcam again

                # We don't want Bartender to start calling people right after serving the client.
                global last_phrase_time
                last_phrase_time = time()
            else:
                cv2.putText(frame, "Please come closer...", (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(APP_WINDOW_TITLE, frame)
                say_random(PHRASES['COME_CLOSER'], 10)

        # Displaying the processed frame
        key = cv2.waitKey(10) & 0xFF # We should give cv2 some time to update the image window.
        cv2.imshow(APP_WINDOW_TITLE, frame)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_loop()
