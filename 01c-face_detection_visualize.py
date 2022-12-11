from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import json
import cv2

base_path = './/train_sample_videos//'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

def draw_facebox(filename, result_list, path):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    ax.set_axis_off()
    if result_list == []:
        return
    result = result_list[0]
    x, y, width, height = result['box']
    rect = plt.Rectangle((x, y), width, height,fill=False, color='orange')
    ax.add_patch(rect)
    for key, value in result['keypoints'].items():
        dot = plt.Circle(value, radius=2, color='red')
        ax.add_patch(dot)
    img_path = os.path.join(path, get_filename_only(filename))
    plt.savefig(img_path,bbox_inches ='tight',pad_inches=0)
    plt.close()
    
detector = MTCNN()

for filename in metadata.keys():
    file_path = os.path.join(base_path, get_filename_only(filename))
    faces_path = os.path.join(file_path, 'faces')
    filtered = [f for f in os.listdir(faces_path) if f[0] != '.']
    new_dir = os.path.join(file_path,"extracted_faces")
    os.makedirs(new_dir, exist_ok=True)
    for path in filtered:
        face_path = os.path.join(faces_path, path)
        image = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image)
        draw_facebox(face_path, faces, new_dir)



