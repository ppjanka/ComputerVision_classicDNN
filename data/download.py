import os
import subprocess

base_dir = os.getcwd()
for type in ["cats", "dogs"]:
    os.chdir(base_dir)
    link_file = base_dir + "/" + type + "_imagenet.txt"
    curr_dir = base_dir + "/" + type
    if not os.path.isdir(curr_dir):
        os.mkdir(curr_dir)
    os.chdir(curr_dir)
    with open(link_file) as f:
        for line in f:
            if len(line) > 1:
                command = "wget " + line.rstrip()
                print('Calling: '+command)
                try:
                    subprocess.run(['wget', line.rstrip()])
                except Exception as e:
                    print('  -- failed: ',e)

os.chdir(base_dir)