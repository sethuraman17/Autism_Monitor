import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import pandas as pd
import sys

root = tk.Tk()
root.title('Attendance System')
root.geometry('640x480')

def register():
    subprocess.run([sys.executable, "register_student.py"])

def attendance():
    subprocess.run([sys.executable, "attendance.py"])

def read_attendance_data():
    attendance_data_path = 'Attendance.csv'
    try:
        data = pd.read_csv(attendance_data_path, parse_dates=['Date'])
        output_text.delete(1.0, tk.END)  # Clear the text area
        output_text.insert(tk.END, data)  # Display the data
        return data
    except FileNotFoundError:
        messagebox.showwarning('Warning', 'Attendance.csv not found.')
        return None
    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')
        return None

def read_register_data():
    register_data_path = 'register_data.csv'
    try:
        data = pd.read_csv(register_data_path)
        output_text.delete(1.0, tk.END)  # Clear the text area
        output_text.insert(tk.END, data)  # Display the data
        return data
    except FileNotFoundError:
        messagebox.showwarning('Warning', 'register_data.csv not found.')
        return None
    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')
        return None

def get_total_registrations():
    data = read_register_data()
    if data is not None:
        total = len(data)
        messagebox.showinfo('Total Registrations', f'Total Registrations: {total}')
        return total
    else:
        return 0

def delete_attendance_record():
    name = name_entry.get()
    date_str = date_entry.get()
    try:
        date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
        attendance_data = read_attendance_data()

        if attendance_data is not None and not attendance_data.empty:
            mask = (attendance_data['Name'] == name) & (attendance_data['Date'] == date)
            attendance_data = attendance_data[~mask]
            attendance_data.to_csv('Attendance.csv', index=False)
            messagebox.showinfo('Success', 'Attendance record deleted successfully.')
        else:
            messagebox.showwarning('Warning', 'No attendance data available or invalid name/date.')

    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')

# Create widgets
buttons_frame = tk.Frame(root)
buttons_frame.pack(fill=tk.X, padx=10, pady=10)

register_button = tk.Button(buttons_frame, text="Register Student", command=register)
register_button.pack(side=tk.LEFT, padx=5)

attendance_button = tk.Button(buttons_frame, text="Mark Attendance", command=attendance)
attendance_button.pack(side=tk.LEFT, padx=5)

read_attendance_button = tk.Button(buttons_frame, text="Attendance Data", command=read_attendance_data)
read_attendance_button.pack(side=tk.LEFT, padx=5)

read_register_button = tk.Button(buttons_frame, text="Registration Data", command=read_register_data)
read_register_button.pack(side=tk.LEFT, padx=5)

read_total_register_button = tk.Button(buttons_frame, text="Total Registration", command=get_total_registrations)
read_total_register_button.pack(side=tk.LEFT, padx=5)

name_label = tk.Label(root, text="Enter Name:")
name_label.pack()

name_entry = tk.Entry(root)
name_entry.pack()

date_label = tk.Label(root, text="Enter Date (YYYY-MM-DD):")
date_label.pack()

date_entry = tk.Entry(root)
date_entry.pack()

delete_button = tk.Button(root, text="Delete Attendance Record", command=delete_attendance_record)
delete_button.pack(pady=10)

output_text = scrolledtext.ScrolledText(root, height=10)
output_text.pack(fill=tk.BOTH, expand=True)

root.mainloop()
