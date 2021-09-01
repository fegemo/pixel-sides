import os
import shutil

def ensure_folder_structure(*folders):
    folder_path = os.getcwd()
    for folder_name in folders:
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)

