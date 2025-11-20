import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the main window

video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
)

if video_path:
    print("Selected video file:", video_path)
    # You can add your video playback code here
else:
    print("No video file selected")

