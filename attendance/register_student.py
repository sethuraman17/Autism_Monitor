import cv2
from tkinter import ttk, simpledialog, messagebox, filedialog, Listbox
import os
import tkinter as tk
from datetime import datetime
import csv
import shutil


class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.vid = cv2.VideoCapture(0)

        self.canvas = tk.Canvas(window, width=self.vid.get(3), height=self.vid.get(4))
        self.canvas.pack()

        self.btn_snapshot = ttk.Button(window, text="Take Photo", command=self.snapshot)
        self.btn_snapshot.pack(padx=5, pady=5)

        self.btn_manage_data = ttk.Button(window, text="Add/Remove Data", command=self.manage_data)
        self.btn_manage_data.pack(padx=5, pady=5)

        self.btn_create_folder = ttk.Button(window, text="Create Folder", command=self.create_folder)
        self.btn_create_folder.pack(padx=5, pady=5)

        self.student_name = ""

        self.photo_folder = "photos"
        if not os.path.exists(self.photo_folder):
            os.makedirs(self.photo_folder)

        # Load the Haar cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.delay = 10
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            if not self.student_name:
                messagebox.showwarning("Warning", "Please create a folder first.")
                return

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                # If faces are detected, proceed to take a photo
                photo_count = len([f for f in os.listdir(os.path.join(self.photo_folder, self.student_name)) if
                                   f.startswith(f"{self.student_name}_img")])
                photo_path = os.path.join(self.photo_folder, self.student_name,
                                          f"{self.student_name}_img{photo_count + 1}.png")
                cv2.imwrite(photo_path, frame)
                messagebox.showinfo("Snapshot", f"Snapshot saved as {photo_path}")
            else:
                # If no faces are detected, show a warning
                messagebox.showwarning("Warning", "No face detected. Try again.")

    def manage_data(self):
        data_window = tk.Toplevel(self.window)
        data_window.title("Manage Data")

        # Listbox to display images
        photo_listbox = Listbox(data_window, selectmode=tk.MULTIPLE)
        photo_listbox.pack(side=tk.LEFT, padx=10, pady=10)
        self.populate_listbox(photo_listbox, self.photo_folder, "Photos:")

        # Buttons to delete selected items
        btn_delete_photo = ttk.Button(data_window, text="Delete Selected Photos",
                                      command=lambda: self.delete_selected_items(photo_listbox, self.photo_folder))
        btn_delete_photo.pack(pady=5)

    def create_folder(self):
        new_student_name = simpledialog.askstring("Input", "Enter student name:")

        if not new_student_name:
            return  # User canceled the input

        existing_folders = [folder for folder in os.listdir(self.photo_folder) if
                            os.path.isdir(os.path.join(self.photo_folder, folder))]
        if new_student_name in existing_folders:
            messagebox.showwarning("Warning", f"The folder for {new_student_name} already exists.")
            return

        self.student_name = new_student_name
        folder_path = os.path.join(self.photo_folder, self.student_name)
        os.makedirs(folder_path, exist_ok=True)
        messagebox.showinfo("Folder Created", f"Folder for {self.student_name} created in {self.photo_folder}.")

        # Save folder creation data to register_data.csv
        self.save_folder_creation_data(new_student_name)

    def save_folder_creation_data(self, student_name):
        csv_file = 'register_data.csv'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Check if the CSV file already exists
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(['Student Name', 'Timestamp'])

        # Append data to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([student_name, timestamp])

    def populate_listbox(self, listbox, folder, label):
        listbox.insert(tk.END, label)
        for file_name in os.listdir(folder):
            listbox.insert(tk.END, file_name)

    def delete_selected_items(self, listbox, folder):
        selected_items = listbox.curselection()
        for index in selected_items:
            if index != 0:  # Skip the label entry
                file_name = listbox.get(index)
                file_path = os.path.join(folder, file_name)

                if os.path.isdir(file_path):
                    # Remove the entire folder
                    shutil.rmtree(file_path)

                    # Extract student name from the folder name
                    student_name = file_name
                    self.delete_student_from_csv(student_name)
                else:
                    # Remove the file
                    os.remove(file_path)

        # Update the listbox after deletion
        listbox.delete(0, tk.END)
        self.populate_listbox(listbox, folder, label="")

    def delete_student_from_csv(self, student_name):
        csv_file = 'register_data.csv'
        temp_file = 'temp_register_data.csv'

        # Read existing data
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = [row for row in reader if row[0] != student_name]

        # Write updated data to a temporary file
        with open(temp_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        # Replace the original CSV file with the updated data
        os.replace(temp_file, csv_file)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1280x720')
    root.resizable(False, False)
    app = VideoApp(root, "Video Capture App")
    root.mainloop()

