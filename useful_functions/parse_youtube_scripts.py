import os

def parse_script(path, path_newfile):
    """
    Description:
    Load the dat file of the youtube script
    the program automatically parse out the time between the lines.
    and connect all the sentences together.
    This will only work for youtube-Transcript-English-English
    return: a newly created file with all time parsed out
    """
    f = open(path, "r")
    f_new = open(path_newfile, "w")
    for line in f:
        if line == "":
            # remove blank line
            continue
        line = line.replace("\n", " ")
        line_list = line.split(":")
        #print(line_list)
        if len(line_list) == 2 and line_list[0].isdigit() and line_list[1].strip(" ").isdigit():
            # remove timestamp
            continue
        f_new.write(line)
    return
