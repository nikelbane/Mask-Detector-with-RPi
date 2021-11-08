# Mask-Detector-with-RPi
## Introduction </br>
With each passing day, technology advancements in the field of electronics, IoT, biomedical sciences have come up with new ideas to fight the SARS Cov2 virus and overcome the pandemic. With each passing week, countries are opening again and trying hard to come back on track. While the danger of getting affected by the virus still persists as it is getting transformed/mutated from one form to another. Hence, the use of proper prevention mechanisms at establishments is the need of the hour.
To address this problem, we are aiming to come up with an integrated assembly of sensors based on a raspberry pi which will check any/all humans for distance, temperature and mask before opening the door.

## Face Mask Detector Assembly </br>
An ML-based program trained over 2000 pictures to detect masks used with a webcam connected to Raspberry Pi will detect the mask un a real-time basis.
This assembly involves building a program to detect whether a person is wearing a mask or not, based on a given image or video stream. It has been accomplished by two main programs, one for training a model based on a given dataset of people wearing masks and people not wearing masks, and the second, by using the model so-trained to detect masks in video streams for real time mask detection. 
For this purpose, we have used python script with Keras and TensorFlow for model processing and training and OpenCV for face detection and image processing. 

![image](https://user-images.githubusercontent.com/54680381/140777372-c59c5154-1ca3-423f-90cb-69519c2bf7a1.png)
	
## Methodology </br>
To begin with, we will break down our project into hardware and software parts each with its own respective sub-steps:
1)	First we will design the circuit by connecting ultrasonic sensor, temperature sensor and web camera on the breadboard along with resistors and LEDs as required.
2)	Then, we will write a python script with the following hardware functionality:
  - Ultrasonic sensor: It will be used for detecting face proximity by calculating distance between face and the sensor module.
  - Temperature sensor: Used for detecting body temperature periodically when face is detected first.
  - Web camera: Used for running face mask detection against the face detected as a result of the ultrasonic sensor by using machine learning algorithm.
3)	We will use the mask detection software built from keras and tensorflow, previously trained and tested from our previous project, for mask recognition along the aforementioned functions.
4)	In the end we will combine the hardware and software for wrapping up the project. 


