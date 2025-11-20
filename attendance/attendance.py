import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3
import imghdr
import time
import pandas as pd

engine = pyttsx3.init()

# Set the voice to a female voice (change the voice ID based on available voices on your system)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

attendance_status = {}

columns = ["Name", "Time"]
data = pd.DataFrame(columns=columns)

# Change the path to the root folder containing subfolders for each person
path = 'photos'
images = []
classNames = []
myList = os.listdir(path)

for person_folder in myList:
    person_path = os.path.join(path, person_folder)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            curImg = cv2.imread(img_path)
            images.append(curImg)
            classNames.append(person_folder)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    global attendance_status, data

    # Get the current date and time as strings
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    # Check if the person has already been marked for attendance today
    attendance_file_path = 'Attendance.csv'
    if os.path.exists(attendance_file_path):
        with open(attendance_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                entry = line.strip().split(',')
                if len(entry) >= 3 and entry[0] == name and entry[2] == current_date:
                    print(f'{name} already marked attendance today.')
                    return

    # If not marked, update the attendance file and status
    with open(attendance_file_path, 'a') as f:
        f.write(f'{name},{current_time},{current_date}\n')

    # Update attendance status
    if current_date not in attendance_status:
        attendance_status[current_date] = []
    attendance_status[current_date].append(name)

    # Speak the welcome message
    engine.say(f'Hi {name}, welcome to Hope Special Needs Center!')
    engine.runAndWait()

    time.sleep(1)

    engine.say('Here is your welcome video, have a nice day')
    engine.runAndWait()

    video_path = f'videos/{name}.mp4'
    if os.path.exists(video_path) and imghdr.what(video_path) is None:
        # play_video(video_path)
        pass
    else:
        print(f'Video file not found for {name}.')

    # Append data to the DataFrame
    data = pd.DataFrame([[name, f'{current_date} {current_time}']], columns=columns)
    
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)
        else:
            name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('a') or key == 27:  
        cv2.destroyAllWindows()
        cap.release()

