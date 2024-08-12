import os

data_folder = '/home/lacie/Github/coopdet3d/data/'
file_path = '/home/lacie/Github/coopdet3d/data/new_val_north_south_v2.txt'

with open(file_path, 'r') as infile:
    with open(data_folder + 'val_north.txt', 'w') as north_file, open(data_folder + 'val_south.txt', 'w') as south_file:
        for line in infile:
            print(line)
            if 'north' in line:
                print('north')
                north_file.write(line.strip() + '\n')
            elif 'south' in line:
                print('south')
                south_file.write(line.strip() + '\n')


