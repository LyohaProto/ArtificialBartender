import cv2
import pyttsx3

APP_WINDOW_TITLE = "Artificial Bartender"

# Configuring text-to-speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 75)

#age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
#gender_list = ['Male', 'Female']
age_list=['baby','child','amateur','very young','young','not so young','old','very old']
gender_list = ['man', 'woman']


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
    faceCascade = cv2.CascadeClassifier('venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    smileCascade = cv2.CascadeClassifier('venv/lib/python3.6/site-packages/cv2/data/haarcascade_smile.xml')

    print('Initializing webcam')
    # grab the reference to the webcam
    webcam = init_webcam()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # capture frames from the camera
    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(200, 200))

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
            cv2.imshow(APP_WINDOW_TITLE, frame)
            cv2.waitKey(100)  # We should give cv2 some time to update the image window.

        if len(faces):
            webcam.release()  # Stopping the webcam, otherwise it keeps capturing frames.

            mood = "happy" if smiles else "sad"
            engine.say(f'Hello, {mood} {age} {gender}')
            if smiles:
                engine.say(f'Would you like some beer?')
            else:
                engine.say(f'Would you like some whiskey?')
            engine.runAndWait()
            key = cv2.waitKey(4000) & 0xFF  # Waiting for the person to go away :)

            webcam = init_webcam()  # Starting the webcam again
        else:
            cv2.putText(frame, "Please come closer...", (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(APP_WINDOW_TITLE, frame)
            key = cv2.waitKey(100) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


def init_webcam():
    # Initializing the webcam
    vs = cv2.VideoCapture(0)
    vs.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH
    vs.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT
    return vs


if __name__ == '__main__':
    capture_loop()
