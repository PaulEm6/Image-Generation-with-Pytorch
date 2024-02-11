import os
import shutil

def empty_repository(repository_path):
    try:
        # Iterate over all files and directories in the repository
        for root, dirs, files in os.walk(repository_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)

        print(f"Repository at '{repository_path}' has been emptied.")
    except Exception as e:
        print(f"Error: {e}")

# Replace 'path/to/your/repository' with the actual path to your repository
repository_path = 'generated'
empty_repository(repository_path)
