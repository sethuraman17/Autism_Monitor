import streamlit as st
import subprocess
import sys

class AutismPatientMonitorApp:
    def __init__(self):
        self.processes = []

    def behaviour(self):
        self.stop_processes()
        process = subprocess.Popen([sys.executable, 'good.py'])
        self.processes.append(process)

    def expression(self):
        self.stop_processes()
        process = subprocess.Popen([sys.executable, 'realtimedetection.py'])
        self.processes.append(process)

    def photo(self):
        self.stop_processes()
        process = subprocess.Popen([sys.executable, 'photo.py'])
        self.processes.append(process)

    def show_data(self):
        data = self.read_behaviour_data()
        st.text_area('Behaviour Data', data)

    def read_behaviour_data(self):
        behaviour_data_path = 'behavior_info.csv'
        try:
            with open(behaviour_data_path, 'r') as file:
                data = file.read()
                return data
        except FileNotFoundError:
            return 'behavior_info.csv not found.'
        except Exception as e:
            return f'An error occurred: {e}'

    def stop_processes(self):
        for process in self.processes:
            process.terminate()

def main():
    st.title("Autism Student Monitor")

    app = AutismPatientMonitorApp()

    behaviour_btn = st.button('Behaviour')
    expression_btn = st.button('Expression')
    photo_btn = st.button('Take Photo')
    show_data_btn = st.button('Show Data')
    stop_btn = st.button('Stop')

    if behaviour_btn:
        app.behaviour()

    if expression_btn:
        app.expression()

    if photo_btn:
        app.photo()

    if show_data_btn:
        app.show_data()

    if stop_btn:
        app.stop_processes()

if __name__ == '__main__':
    main()
