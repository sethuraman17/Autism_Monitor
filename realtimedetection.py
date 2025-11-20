import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

confidence_threshold = 0.5  # Set your confidence threshold here

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            face_roi = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            face_roi_resized = cv2.resize(face_roi, (48, 48))
            img = extract_features(face_roi_resized)
            pred = model.predict(img)

            # Get the predicted label and confidence
            prediction_label = labels[pred.argmax()]
            confidence = np.max(pred)

            # Check if the prediction confidence is above the threshold
            if confidence > confidence_threshold:
                cv2.putText(im, f'{prediction_label} ({confidence:.2f})', (p - 10, q - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        cv2.imshow("Output", im)
        k = cv2.waitKey(27)
        if k == ord('q'):
            break

    except cv2.error:
        pass
webcam.release()
cv2.destroyAllWindows()
