import json
import os
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import splitfolders

base_path = './/train_sample_videos//'
dataset_path = './/prepared_dataset//'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = './/tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)
os.makedirs(tmp_fake_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)
for filename in metadata.keys():
    print(filename)
    print(metadata[filename]['label'])
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')
    print(tmp_path)
    if os.path.exists(tmp_path):
        if metadata[filename]['label'] == 'REAL':    
            print('Copying to :' + real_path)
            copy_tree(tmp_path, real_path)
        elif metadata[filename]['label'] == 'FAKE':
            print('Copying to :' + tmp_fake_path)
            copy_tree(tmp_path, tmp_fake_path)
        else:
            print('Ignored..')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))
all_real_faces = sorted(all_real_faces)
all_fake_faces = [f for f in sorted(os.listdir(tmp_fake_path)) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))



random_faces = all_fake_faces[0:len(all_real_faces)]
for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print('Down-sampling Done!')



test_val_size = int(0.1 * len(all_real_faces))
test_val_tgth = 2 * test_val_size
train_size = len(all_real_faces) - test_val_tgth

split_path = './/split_dataset//'
print('Creating Directory: ' + split_path)
os.makedirs(split_path, exist_ok=True)

train_path = os.path.join(split_path, 'train')
test_path = os.path.join(split_path, 'test')
val_path = os.path.join(split_path, 'val')

print('Creating Directory: ' + train_path)
os.makedirs(train_path, exist_ok=True)
train_real = os.path.join(train_path, 'real')
train_fake = os.path.join(train_path, 'fake')
print('Creating Directory: ' + train_real)
os.makedirs(train_real, exist_ok=True)
print('Creating Directory: ' + train_fake)
os.makedirs(train_fake, exist_ok=True)

print('Creating Directory: ' + test_path)
os.makedirs(test_path, exist_ok=True)
test_real = os.path.join(test_path, 'real')
test_fake = os.path.join(test_path, 'fake')
print('Creating Directory: ' + test_real)
os.makedirs(test_real, exist_ok=True)
print('Creating Directory: ' + test_fake)
os.makedirs(test_fake, exist_ok=True)

print('Creating Directory: ' + val_path)
os.makedirs(val_path, exist_ok=True)
val_real = os.path.join(val_path, 'real')
val_fake = os.path.join(val_path, 'fake')
print('Creating Directory: ' + val_real)
os.makedirs(val_real, exist_ok=True)
print('Creating Directory: ' + val_fake)
os.makedirs(val_fake, exist_ok=True)



# REFERNECE: dataset_path = './/prepared_dataset//' and then theres real and fake
# move from prepared_dataset/real or fake to 
# REFERNECE: real_path = os.path.join(dataset_path, 'real')
for i in range(len(all_real_faces)):
    real_file_name = all_real_faces[i]
    fake_file_name = all_fake_faces[i]
    real_src = os.path.join(real_path, real_file_name)
    fake_src = os.path.join(fake_path, fake_file_name)

    if i > test_val_tgth:
        # then it's training
        real_dest = os.path.join(train_real, real_file_name)
        fake_dest = os.path.join(train_fake, fake_file_name)
        shutil.copyfile(real_src, real_dest)
        shutil.copyfile(fake_src, fake_dest)
    elif i > test_val_size:
        # then it's val
        real_dest = os.path.join(test_real, real_file_name)
        fake_dest = os.path.join(test_fake, fake_file_name)
        shutil.copyfile(real_src, real_dest)
        shutil.copyfile(fake_src, fake_dest)
    else:
        # it's testing
        real_dest = os.path.join(val_real, real_file_name)
        fake_dest = os.path.join(val_fake, fake_file_name)
        shutil.copyfile(real_src, real_dest)
        shutil.copyfile(fake_src, fake_dest)
print("Train/test/val split done!")