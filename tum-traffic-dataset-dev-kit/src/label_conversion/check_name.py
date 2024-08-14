import os

def read_file_to_list(file_path):
    """Reads a file and returns a list of lines."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def list_files_in_folder(folder_path):
    """Lists all files in a folder."""
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Paths to the two files and the folder
file1_path = '/home/lacie/Github/tum-traffic-dataset-dev-kit/data_split/train/images/s110_camera_basler_south2_8mm/file_names.txt'
file2_path = '/home/lacie/Github/tum-traffic-dataset-dev-kit/data_split/val/images/s110_camera_basler_south2_8mm/file_names.txt'
folder_paths = [
    '/mnt/Data-2/Datasets/A9_dataset_all/A9_dataset_origin/R02/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm',
    '/mnt/Data-2/Datasets/A9_dataset_all/A9_dataset_origin/R02/a9_dataset_r02_s02/images/s110_camera_basler_south2_8mm',
    '/mnt/Data-2/Datasets/A9_dataset_all/A9_dataset_origin/R02/a9_dataset_r02_s03/images/s110_camera_basler_south2_8mm',
    '/mnt/Data-2/Datasets/A9_dataset_all/A9_dataset_origin/R02/a9_dataset_r02_s04/images/s110_camera_basler_south2_8mm',
]

# Read the contents of the two files
file1_contents = read_file_to_list(file1_path)
file2_contents = read_file_to_list(file2_path)

# Combine the contents of the two files into a single set for faster lookup
combined_file_contents = set(file1_contents + file2_contents)

# Collect all files from the specified folders
all_folder_files = set()
for folder_path in folder_paths:
    folder_files = list_files_in_folder(folder_path)
    all_folder_files.update(folder_files)

# Find files in the combined folder files that are not in the combined file contents
files_not_in_files = [f for f in all_folder_files if f not in combined_file_contents]

# Print the result
print("Files in the folders not included in the two files:")
for file in files_not_in_files:
    print(file)