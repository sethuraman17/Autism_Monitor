import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import subprocess
import sys

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Autism Patient Monitor')
        self.root.geometry('320x240')

        self.behaviour_btn = tk.Button(root, text='Behaviour', command=self.open_behaviour_window)
        self.expression_btn = tk.Button(root, text='Expression', command=self.expression)
        self.photo_btn = tk.Button(root, text='Take Photo', command=self.photo)
        self.show_data_btn = tk.Button(root, text='Show Data', command=self.show_data)
        self.stop_btn = tk.Button(root, text='Stop', command=self.stop_processes)

        self.behaviour_btn.pack(pady=5)
        self.expression_btn.pack(pady=5)
        self.photo_btn.pack(pady=5)
        self.show_data_btn.pack(pady=5)
        self.stop_btn.pack(pady=5)

        self.processes = []

    def open_behaviour_window(self):
        behaviour_window = tk.Toplevel(self.root)
        behaviour_window.title('Behaviour Options')

        live_btn = tk.Button(behaviour_window, text='Live', command=self.run_live_behaviour)
        upload_btn = tk.Button(behaviour_window, text='Upload Video', command=self.run_upload_behaviour)

        live_btn.pack(pady=5)
        upload_btn.pack(pady=5)

    def run_live_behaviour(self):
        process = subprocess.Popen([sys.executable, 'good.py'])
        self.processes.append(process)

    def run_upload_behaviour(self):
        process = subprocess.Popen([sys.executable, 'try.py'])
        self.processes.append(process)

    def expression(self):
        process = subprocess.Popen([sys.executable, 'realtimedetection.py'])
        self.processes.append(process)

    def photo(self):
        process = subprocess.Popen([sys.executable, 'photo.py'])
        self.processes.append(process)

    def show_data(self):
        data = self.read_behaviour_data()
        self.display_data(data, 'Behaviour Data')

    def read_behaviour_data(self):
        behaviour_data_path = 'behavior_info.csv'
        data = ''
        try:
            with open(behaviour_data_path, 'r') as file:
                data = file.read()
        except FileNotFoundError:
            tk.messagebox.showinfo('File Not Found', 'behavior_info.csv not found.')
        except Exception as e:
            tk.messagebox.showerror('Error', f'An error occurred: {e}')
        return data

    def display_data(self, data, title):
        data_window = tk.Toplevel(self.root)
        data_window.title(title)

        text_widget = scrolledtext.ScrolledText(data_window, wrap=tk.WORD, width=40, height=20)
        text_widget.pack(padx=10, pady=10)

        text_widget.insert(tk.END, data)

    def stop_processes(self):
        for process in self.processes:
            process.terminate()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
