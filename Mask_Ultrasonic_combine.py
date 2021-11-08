from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

# GPIO Pin setup
TRIG = 16
ECHO = 18
LED = 8
i=0

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(LED, GPIO.OUT, initial=GPIO.LOW)

GPIO.output(TRIG, False)

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []	#List of faces
	locs = []	#Corresponding locations
	preds = []	#List of predictions


	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) 
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"/home/pi/Minor-project/face_detector/deploy.prototxt"
weightsPath = r"/home/pi/Minor-project/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("/home/pi/Minor-project/mask_detector.h5")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

try:
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        label1 = ""

        # detect faces in the frame and predict if they are wearing a face mask
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label1 = label
            # include the probability in the label and display it
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if label1 == "Mask":
            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)

            while GPIO.input(ECHO)==0:
                pulse_start = time.time()

            while GPIO.input(ECHO)==1:
                pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150
            distance = round(distance+1.15, 2)
  
            if distance<=20 and distance>=5:
                GPIO.output(8, GPIO.HIGH)
                time.sleep(3) 
                GPIO.output(8, GPIO.LOW) 
                time.sleep(1)
                print ("Open- distance:",distance,"cm")
                i=1
          
            if distance>20 and i==1:
                print ("Now closed....")
                i=0
            
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

except KeyboardInterrupt:
    GPIO.cleanup()

GPIO.cleanup()
cv2.destroyAllWindows()
vs.stop()