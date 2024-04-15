import os
import json

def pair_images_with_labels(data_dir):
  image_label_pairs = {}
  for root, _, files in os.walk(data_dir):
    if root == data_dir:
      continue
    # Extract label from folder name
    label = os.path.basename(root)
    ##########
    if(label[-2:]!='_r'):
      continue
      label=label[:-2]
    ##########
    for filename in files:
      # Check for image extensions (adjust as needed)
      if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(root, filename)
        image_label_pairs[image_path]=label
  return image_label_pairs

data_dir = "./dog_v1"  
image_label_pairs = pair_images_with_labels(data_dir)

with open("data&labels.json", "w") as f:
  json.dump(image_label_pairs, f, indent=4)