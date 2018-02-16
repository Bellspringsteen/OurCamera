r"""Download images from NYC DOT

This executable is used to download DOT images from http://dotsignals.org/:

By inspecting http://dotsignals.org/ you can determine the URL of the camera you are interested in for example
http://207.251.86.238/cctv476.jpg?math=0.29640904121642864

* the url's seem to be static over months.
* DOT does seem to turn the cameras occasionally i.e. move it to cover a different street


Example usage:
    ./download_dot_files \
        --url=http://207.251.86.238/cctv476.jpg?math=0.29640904121642864 \
        --save_directory=path/to/data_dir
"""

import urllib
import threading
import datetime
import argparse
from argparse import RawTextHelpFormatter


def download_dot_files(args):
    now = datetime.datetime.now()
    try:
        urllib.urlretrieve(args.url, args.save_directory+str(now)+".jpg")
        print("Saved file to "+args.save_directory+str(now)+".jpg")
    except IOError:
        pass

    t = threading.Timer(1.0, download_dot_files,[args]).start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download images every second from dotsignals.org', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-url', help='the url for the image you want to download')
    parser.add_argument('-save_directory', help='the directory you want to save the images to')
    args = parser.parse_args()
    download_dot_files(args)
