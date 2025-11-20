import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import filedialog
import os

class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.vid = cv2.VideoCapture(0)

        self.canvas = tk.Canvas(window, width=self.vid.get(3), height=self.vid.get(4))
        self.canvas.pack()

        self.btn_snapshot = ttk.Button(window, text="Take Photo", command=self.snapshot)
        self.btn_snapshot.pack(padx=10, pady=10)

        self.btn_manage_data = ttk.Button(window, text="Add/Remove Data", command=self.manage_data)
        self.btn_manage_data.pack(padx=10, pady=10)

        self.photo_folder = "photos"
        if not os.path.exists(self.photo_folder):
            os.makedirs(self.photo_folder)

        self.delay = 10
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            user_input = simpledialog.askstring("Input", "Enter file name:")
            if user_input:
                photo_path = os.path.join(self.photo_folder, f"{user_input}.png")
                cv2.imwrite(photo_path, frame)
                messagebox.showinfo("Snapshot", f"Snapshot saved as {photo_path}")

    def manage_data(self):
        data_window = tk.Toplevel(self.window)
        data_window.title("Manage Data")

        # Listbox to display images
        photo_listbox = tk.Listbox(data_window, selectmode=tk.MULTIPLE)
        photo_listbox.pack(side=tk.LEFT, padx=10, pady=10)
        self.populate_listbox(photo_listbox, self.photo_folder, "Photos:")

        # Button to delete selected items
        btn_delete_photo = ttk.Button(data_window, text="Delete Selected Photos", command=lambda: self.delete_selected_items(photo_listbox))
        btn_delete_photo.pack(pady=5)

    def populate_listbox(self, listbox, folder, label):
        listbox.insert(tk.END, label)
        for file_name in os.listdir(folder):
            listbox.insert(tk.END, file_name)

    def delete_selected_items(self, listbox):
        selected_items = listbox.curselection()
        for index in selected_items:
            if index != 0:  # Skip the label entry
                file_name = listbox.get(index)
                file_path = os.path.join(self.photo_folder, file_name)
                os.remove(file_path)
        # Update the listbox after deletion
        listbox.delete(0, tk.END)
        self.populate_listbox(listbox, self.photo_folder, label="")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the VideoApp class
root = tk.Tk()
app = VideoApp(root, "Video Capture App")
