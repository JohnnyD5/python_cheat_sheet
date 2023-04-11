from parse_youtube_scripts import parse_script
import time
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    print(__file__)
    #print(os.path.join(os.path.dirname(__file__), '..'))
    #print(os.path.dirname(os.path.realpath(__file__)))
    #print(os.path.abspath(os.path.dirname(__file__)))
    path = "test.txt"
    path_newfile = "test_new.txt"
    parse_script(path, path_newfile)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
