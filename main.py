#!/usr/bin/env python3

import random
import textwrap
import tkinter
from time import sleep, time
from tkinter import *
from tkinter import messagebox

import pyttsx3

from image_processor import ImageProcessor
import cv2
from PIL import ImageTk, Image, ImageDraw, ImageFont

WEBCAM_RES_W = 800
WEBCAM_RES_H = 600
FACE_TRIGGER_WIDTH = 150
ATTRACT_CUSTOMERS_INTERVAL = 15
TEXT_MAX_LENGTH = 80

PHRASES = {
    'COME_CLOSER': [
        'Hey, I can see you! Come closer!',
        "Don't be shy! Come closer and I'll choose a drink for you.",
        'Hey! Come here!',
        "Don't be afraid, human! A robot may not injure a human being or, through inaction, allow a human being to come to harm."
    ],
    'GREETINGS': {
        'FORMAL': [
            'Bon jour',
            'Greetings',
            'Good evening',
            'Hello'
        ],
        'INFORMAL': [
            'Hi',
            "What's up",
            'Hey hey hey'
        ]
    },
    'PROMPTS': {
        'FEMALE_INFORMAL': [
            'beautiful',
            'gorgeous',
            'female human'
        ],
        'FEMALE_FORMAL': [
            'madame'
        ],
        'MALE_INFORMAL': [
            'dude',
            'fella',
            'buddy',
            'bro',
            'kiddo',
            'mate'
        ],
        'MALE_FORMAL': [
            'mister',
            'sir',
            'male human'
        ]
    },
    'PHRASES': {
        'FEMALE_INFORMAL': [
            "You look so good that I'll have to ask you for an ID.",
            "Wow, you look amazing, can I get you something to drink?",
            "It's Friday, you made it! Can I offer you a drink?",
            "Did the sun just come out or did you just smile at me? Let me buy you a drink.",
            "You are so beautiful that you made me forget my pickup line. How about a drink instead?",
            "I have one question. Wine not?"
        ],
        'FEMALE_FORMAL': [
            'Do you still get asked for ID?',
            "Looking for something to drink?",
            "It's Friday, you made it! Can I offer you a drink?",
            "Happy Friday! Can I get you something to drink?"
        ],
        'MALE_INFORMAL': [
            "Looking for something to drink?",
            "Lookin' for a drink to relax after a busy day?",
            "It's Friday, you made it! Can I offer you a drink?",
            "Happy Friday! Can I get you something to drink?"
        ],
        'MALE_FORMAL': [
            'Are you sure you are old enough to get a drink?',
            "You look so good that I'll have to ask you for an ID. Just kidding. I'm a robot, you know...",
            'Do you still get asked for ID?',
            "It's Friday, you made it! Can I offer you a drink?"
        ]
    },
    'WHY_SO_SAD': [
        'Why so serious?',
        "I'm sure you need something to cheer up!",
        "You look like you could use a drink... or two.",
        "The answer may not lie at the bottom of a beer bottle, but you should at least check!",
        "You know what rhymes with Friday? Alcohol! Get some!",
        "Wine is not the answer. Wine is the question. Yes, is the answer.",
        "I'm doing my part to conserve water by drinking beer instead, you should try it too."
    ],
    'DRINKS': [
        'Heineken',
        'Cider',
        'Heineken Light',
        'Sauvignon Blanc',
        'Chardonnay',
        'Pinot Gris',
        'Rose',
        'Pinot Noir'
    ]
}


class GUI:
    def __init__(self, webcam_w=800, webcam_h=600, min_face_width=200):
        self.root = tkinter.Tk()
        self.root.title("Artificial Bartender")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Configure>", self.on_resize)

        self.vid_label = Label(self.root, anchor="center", background='black', width=webcam_w, height=webcam_h)
        self.vid_label.pack(expand=YES, fill=BOTH)

        self.window_width = self.webcam_w = webcam_w
        self.window_height = self.webcam_h = webcam_h
        self.image_ratio = float(webcam_w / webcam_h)
        self.min_face_width = min_face_width

        self.cap = None
        self.start_webcam()

        # Configuring text-to-speech
        self.tts_engine = pyttsx3.init()
        rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', rate - 60)

        self.image_processor = ImageProcessor('font.ttf')
        self.text_font_size = 20
        self.text_font = ImageFont.truetype('font.ttf', self.text_font_size)
        self.last_phrase_time = time()

        self.process_webcam()
        self.root.mainloop()

    def process_webcam(self):
        ret, frame = self.cap.read()

        faces = self.image_processor.detect_faces(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        self.draw_face_rectangles(faces, frame)

        # Resizing the whole image
        if self.window_height > self.webcam_h:
            frame = frame.resize((int(self.window_height * self.image_ratio), self.window_height))

        # Displaying the image on the form
        self.update_tk_image(frame)

        # Checking if there's a customer
        served = self.check_for_customers(faces)

        # Attract customers
        if not served and len(faces):
            self.attract_customers(ATTRACT_CUSTOMERS_INTERVAL)

        self.vid_label.after(10, self.process_webcam)

    def check_for_customers(self, faces):
        for face in faces:
            if face['xy'][2] >= self.min_face_width:
                random.seed(time())

                if face['age'] in ['AGE_0-2', 'AGE_4-6', 'AGE_8-12', 'AGE_15-20', 'AGE_25-32']:
                    style = 'INFORMAL'
                else:
                    style = 'FORMAL'

                text = textwrap.fill(random.choice(PHRASES["GREETINGS"][style]) + " " + \
                                     random.choice(PHRASES['PROMPTS'][f"{face['gender']}_{style}"]), TEXT_MAX_LENGTH) + "!\n"
                if style == 'INFORMAL' and face['smile'] is False:
                    text += textwrap.fill(random.choice(PHRASES['WHY_SO_SAD']), TEXT_MAX_LENGTH) + "\n"
                text += textwrap.fill(random.choice(PHRASES['PHRASES'][f"{face['gender']}_{style}"]), TEXT_MAX_LENGTH) + "\n"
                text += textwrap.fill(f"Would you like some {random.choice(PHRASES['DRINKS'])}?", TEXT_MAX_LENGTH)

                self.say_and_display_text(text)

                self.last_phrase_time = time()
                return True
        return False

    def attract_customers(self, interval=10):
        if time() - self.last_phrase_time > interval:
            random.seed(time())
            self.say_and_display_text(textwrap.fill(random.choice(PHRASES['COME_CLOSER']), TEXT_MAX_LENGTH))
            self.last_phrase_time = time()

    def say_and_display_text(self, text):
        # Stopping the webcam, otherwise it will continue capturing frames
        self.stop_webcam()
        sleep(0.2)
        # Filling the screen with black, displaying the text and starting the tts engine
        black_image = Image.new('RGB', (self.webcam_w, self.webcam_h))
        image_text = ImageDraw.Draw(black_image)
        image_text.multiline_text((10, 10), text, font=self.text_font, fill=(0, 255, 0, 128))

        # Resizing the whole image
        if self.window_height > self.webcam_h:
            black_image = black_image.resize((int(self.window_height * self.image_ratio), self.window_height))

        # Displaying the image on the form
        self.update_tk_image(black_image)
        # Starting the TTS
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        # Delay before returning back to webcam frames analysis loop
        self.start_webcam()

    def update_tk_image(self, frame):
        image = ImageTk.PhotoImage(frame)
        self.vid_label.configure(image=image)
        self.vid_label.image = image
        self.root.update()

    def draw_face_rectangles(self, faces, frame):
        # Drawing rectangles around the faces
        image_text = ImageDraw.Draw(frame)
        for face in faces:
            x, y, w, h = face['xy']
            trigger_line_coeff = int((self.min_face_width - w) / 2)

            if trigger_line_coeff > 0:
                face_rectangle_color = (0, 255, 0)

                image_text.line((x - trigger_line_coeff, y, x - trigger_line_coeff, y + w), fill=(255, 0, 0))
                image_text.line((x + w + trigger_line_coeff, y, x + w + trigger_line_coeff, y + w), fill=(255, 0, 0))

                face_annotation = f"{face['gender']}, {face['age']}"
                text_pos_x = x + int((w - image_text.textsize(face_annotation, self.text_font))[0] / 2)
                image_text.text(
                    (text_pos_x, y - self.text_font_size - 5),
                    face_annotation,
                    font=self.text_font,
                    fill=(0, 255, 0, 128)
                )

                face_hint = "Get your face between the red lines"
                hint_pos_x = x + int((w - image_text.textsize(face_hint, self.text_font))[0] / 2)
                image_text.text(
                    (hint_pos_x, y + h + self.text_font_size + 5),
                    face_hint,
                    font=self.text_font,
                    fill=(255, 0, 0, 128)
                )
            else:
                face_rectangle_color = (255, 0, 0)

            image_text.rectangle([x, y, x + w, y + h], outline=face_rectangle_color)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cap.release()
            self.root.destroy()

    def on_resize(self, event):
        self.window_width = int(event.width) - 2
        self.window_height = int(event.height) - 2

    def start_webcam(self, webcam_number=0):
        # Initializing the webcam
        vs = cv2.VideoCapture(webcam_number)
        vs.set(3, self.webcam_w)  # CV_CAP_PROP_FRAME_WIDTH
        vs.set(4, self.webcam_h)  # CV_CAP_PROP_FRAME_HEIGHT

        ret = False
        retry = 0
        while not ret:
            ret, frame = vs.read()
            sleep(0.1)
            retry += 1
            assert retry < 30, 'Failed to initialize webcam'

        self.cap = vs

    def stop_webcam(self):
        self.cap.release()


App = GUI(WEBCAM_RES_W, WEBCAM_RES_H, FACE_TRIGGER_WIDTH)
