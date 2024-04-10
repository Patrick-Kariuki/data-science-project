from rembg import remove
from PIL import Image
import os
from alive_progress import alive_bar; import time
#folds = ['french_bulldog','german_shepherd','golden_retriever','poodle','yorkshire_terrier']
# folds = ['french_bulldog']
# for each in folds:
#     folder_path= f"./dog_v1/{each}"
#     for file in os.listdir(folder_path):
#         image_path = os.path.join(folder_path, file)
#         img = Image.open(image_path)
#         output=remove(img)
#         output=output.convert('RGB')
#         new = folder_path+'_r/'+file
#         if not os.path.exists(folder_path+'_r/'):
#             os.makedirs(folder_path+'_r/')
#         with open(new,'w') as file:
#             file.write(".")
#         output.save(new,format='JPEG')
#         print(new)