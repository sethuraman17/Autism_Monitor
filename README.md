# Autism Student Monitor

## Demo Intro

https://github.com/user-attachments/assets/36daa274-883a-4a44-a310-31f5f42563f1

https://github.com/user-attachments/assets/6707b996-5d45-46d5-9650-9b76ee365bf1

https://github.com/user-attachments/assets/a6c14d8f-ffc6-4c77-8697-1ca1b80c956a

## Project Overview

This project is a comprehensive system designed to assist in monitoring and understanding the behavior and emotional state of students, with a particular focus on applications for students with autism. It uses computer vision and machine learning to perform real-time analysis of facial expressions and physical behaviors.

The application provides a user-friendly interface to access its various features, and it can be run through a web interface (Streamlit) or a desktop GUI (Tkinter).

## Features

*   **Behavior Analysis**: Identifies and logs various behaviors such as "SittingInChair&Active", "Bitting", "Fighting", "Running", "SittingInChair&IN-Active", and "HandFlapping".
*   **Facial Emotion Recognition**: Detects and recognizes facial expressions like 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', and 'surprise'.
*   **Real-time Monitoring**: Performs behavior and emotion analysis in real-time using a webcam.
*   **Video Upload**: Allows users to upload video files for behavior analysis.
*   **Data Logging**: Saves the results of the behavior analysis to a CSV file (`behavior_info.csv`).
*   **Data Visualization**: Generates a pie chart to visualize the distribution of different behaviors.
*   **Face Recognition**: Can identify individuals by name if their photos are provided.
*   **Photo Management**: A tool to capture and manage photos used for face recognition.

## For Users

### Running the Application

There are two ways to run the application:

1.  **Web Interface (Streamlit)**: This is the recommended way to run the application.
2.  **Desktop GUI (Tkinter)**: A desktop-based interface.

#### Prerequisites

Before you begin, ensure you have Python installed on your system. You will also need to install the required libraries.

```bash
pip install -r requirements.txt
```

#### Running the Web Interface

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the application's user interface.

#### Running the Desktop GUI

To run the Tkinter application, execute the following command in your terminal:

```bash
python gui.py
```

This will open a desktop window with the application's user interface.

## For Developers

### Project Structure

```
.
├── attendance/
├── photos/
├── .gitignore
├── Dockerfile
├── app.py              # Main application file (Streamlit)
├── good.py             # Real-time behavior analysis
├── realtimedetection.py  # Real-time facial emotion recognition
├── photo.py            # Photo management application (Tkinter)
├── gui.py              # Desktop GUI application (Tkinter)
├── try.py              # Behavior analysis for uploaded videos
├── requirements.txt    # Project dependencies
├── model.h5            # Behavior analysis model
├── facialemotionmodel.h5 # Facial emotion recognition model
├── yolov5su.pt         # YOLOv5 model
└── yolov8n.pt          # YOLOv8 model
```

### How it Works

The application is modular, with different scripts responsible for specific tasks:

*   **`app.py`** and **`gui.py`** serve as the main entry points for the user. They provide a simple interface to launch the different modules.
*   **`good.py`** is the core of the behavior analysis module. It uses the YOLO model for person detection, the SORT algorithm for tracking, a custom Keras model (`model.h5`) for behavior classification, and the `face_recognition` library to identify individuals.
*   **`realtimedetection.py`** handles facial emotion recognition. It uses a Keras model (`facialemotionmodel.h5`) and Haar cascades for face detection.
*   **`photo.py`** is a utility for capturing and managing photos of individuals. These photos are stored in the `photos/` directory and are used by `good.py` for face recognition.
*   **`try.py`** is similar to `good.py` but is designed to work with video files instead of a live webcam feed.

### Dependencies

The project has a number of dependencies, which are listed in `requirements.txt`. The key libraries are:

*   **Streamlit**: For the web interface.
*   **Tkinter**: For the desktop GUI.
*   **OpenCV**: For video capture and image processing.
*   **TensorFlow/Keras**: For running the machine learning models.
*   **face-recognition**: For face recognition.
*   **ultralytics (YOLO)**: For object detection.
*   **Pandas**: For data manipulation and logging.
*   **Matplotlib**: For data visualization.

### Models

The project uses several pre-trained models:

*   **`model.h5`**: A Keras model for behavior classification.
*   **`facialemotionmodel.h5`**: A Keras model for facial emotion recognition.
*   **`yolov5su.pt` and `yolov8n.pt`**: YOLO models for object detection.
*   **`best.h5`**: Another model file, purpose to be determined.

### Docker

A `Dockerfile` is provided to containerize the application. To build the Docker image, run:

```bash
docker build -t autism-student-monitor .
```

To run the application in a Docker container:

```bash

docker run -p 8501:8501 autism-student-monitor
```

# Input Video

https://github.com/user-attachments/assets/c989d797-bcaf-4eed-b99e-3a2f1c205470

https://github.com/user-attachments/assets/1f69289b-16ba-43e5-ae21-4140a8f1fa0d
