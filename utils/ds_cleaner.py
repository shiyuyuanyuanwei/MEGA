# author:c19h
# datetime:2022/12/26 22:03
import os

path = '../'
for root, dirs, files in os.walk(path):
    for file in files:
        if file == ".DS_Store":
            path = os.path.join(root, file)
            print(path)
            os.remove(path)
