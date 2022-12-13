import math
import os
import matplotlib.pyplot as plt

# Config:
images_dir = './complete_data/face_marked_grid/'
result_grid_filename = './complete_data/face_marked_frames.jpg'
result_figsize_resolution = 40 # 1 = 100px

images_list = sorted(os.listdir(images_dir))
filtered = []
for an_img in images_list:
    if an_img[0] != '.':
        filtered.append(an_img)
images_count = len(filtered)
print('Images: ', filtered)
print('Images count: ', images_count)

# Calculate the grid size:
grid_size = math.ceil(math.sqrt(images_count))

# Create plt plot:
fig, axes = plt.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))
fig.subplots_adjust(wspace=0, hspace=0)

current_file_number = 0
for image_filename in filtered:
    x_position = current_file_number // grid_size
    y_position = current_file_number % grid_size
    if image_filename[0] == '.':
        print("not this")
    else:
        plt_image = plt.imread(images_dir + '/' + filtered[current_file_number])
        axes[x_position, y_position].imshow(plt_image)
        print((current_file_number + 1), '/', images_count, ': ', image_filename)

        current_file_number += 1

plt.subplots_adjust(left=0.0, right=0.5, bottom=0.0, top=0.5)
plt.savefig(result_grid_filename)
