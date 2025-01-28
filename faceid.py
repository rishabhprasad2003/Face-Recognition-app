# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Building app and layer
class CamApp(App):

    def build(self):
        # Main Layout components
        self.web_cam = Image(size_hint = (1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiator", size_hint=(1, .1))


        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):

        #Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[200-125:200+125, 300-125:300+125, :]

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and convert to 100x100 pixels
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100,100))
        img = img/255.0
        return img

    # to verify a person
    def verify(self, *args):
        detection_threshold = 0.6
        verification_threshold = 0.6

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[200-125:200+125, 300-125:300+125, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
            results.append(result)

        
        detection = np.sum(np.array(results)>detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        #Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        # Logger.info(np.sum(np.array(results)>0.2))
        # Logger.info(np.sum(np.array(results)>0.4))
        # Logger.info(np.sum(np.array(results)>0.5))
        # Logger.info(np.sum(np.array(results)>0.8))


        return results, verified

if __name__ == '__main__':
    CamApp().run()