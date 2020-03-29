import os

from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from shutil import copyfile

source_folder_1 = './original_seq/youtube/'
source_folder_2 = './manipulated_seq/NeuralTexture/'
dest_folder_1 = './train/1'
dest_folder_2 = './train/0'
dest_folder_3 = './test/1'
dest_folder_4 = './test/0'


for i, j in zip(listdir(source_folder_1)[:860], listdir(source_folder_2)[:860]):
    copyfile(join(source_folder_1, i), join(dest_folder_1, i))
    copyfile(join(source_folder_2, j), join(dest_folder_2, j))

for i, j in zip(listdir(source_folder_1)[860:], listdir(source_folder_2)[860:]):
    copyfile(join(source_folder_1, i), join(dest_folder_3, i))
    copyfile(join(source_folder_2, j), join(dest_folder_4, j))