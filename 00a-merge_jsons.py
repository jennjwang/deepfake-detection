import json

json_str1 = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/train_sample_videos/metadata.json"
json_str2 = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/02_second_train_sample_videos/metadata.json"
with open(json_str1) as json_file1:
    dict1 = json.load(json_file1)

with open(json_str2) as json_file2:
    dict2 = json.load(json_file2)

merged_dict = {key: value for (key, value) in (dict1.items() + dict2.items())}
jsonString_merged = json.dumps(merged_dict)

with open("/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/combined.json", "w") as outfile:
    json.dump(merged_dict, outfile)

