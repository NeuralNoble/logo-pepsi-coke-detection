import os
def process_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            with open(file_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == '0':
                        parts[0] = '1'
                    file.write(" ".join(parts) + "\n")


# Example usage
folder_path = '/Users/amananand/PycharmProjects/logo_detection/data_preparation/labels'  # Replace with the path to your folder
process_files(folder_path)
print('Done')
