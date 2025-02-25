# main.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from gui import start_gui

if __name__ == "__main__":
    start_gui()