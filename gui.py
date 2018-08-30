import random
import tkinter
from time import sleep, time
from tkinter import *
from tkinter import messagebox

import pyttsx3

from image_processor import ImageProcessor
import cv2
from PIL import ImageTk

WEBCAM_RES_W = 800
WEBCAM_RES_H = 600

class GUI:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.vid_label = Label(self.root, anchor=NW)
        self.vid_label.pack(expand=YES, fill=BOTH)

        self.cap = self.init_webcam()

        # Configuring text-to-speech
        self.tts_engine = pyttsx3.init()
        rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', rate - 60)

        self.image_processor = ImageProcessor('font.ttf')

        self.process_webcam()
        self.root.mainloop()

    def process_webcam(self):
        ret, frame = self.cap.read()

        processed_frame, text = self.image_processor.process_frame(frame, self.root.winfo_width(), self.root.winfo_height())

        image = ImageTk.PhotoImage(processed_frame)
        self.vid_label.configure(image=image)
        self.vid_label.image = image

        if text:
            self.cap.release()
            self.root.update()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            sleep(1)
            self.cap = self.init_webcam()

        self.vid_label.after(10, self.process_webcam)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cap.release()
            self.root.destroy()

    def init_webcam(self):
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

App = GUI()
