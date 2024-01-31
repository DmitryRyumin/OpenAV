import os
import subprocess


def convert_webm_to_mp4(input_path, output_path):
    line = ('ffmpeg -i ' + os.getcwd() + "/" + input_path
            + " " + os.getcwd() + "/" + output_path)
    subprocess.run(line)