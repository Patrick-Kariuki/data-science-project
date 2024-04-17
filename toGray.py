from PIL import Image
from rembg import remove
import os
from alive_progress import alive_bar
folds = ['french_bulldog_r','german_shepherd_r','golden_retriever_r','poodle_r','yorkshire_terrier_r']
for each in folds:
    print(f"working on {each} folder")
    folder_path= f"./dog_v1/{each}"
    fileLen=len([entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))])
    with alive_bar(fileLen) as bar:
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            img = img = Image.open(image_path).convert('L')
            output=img
            #output=output.convert('RGB')
            new = f'./dog_v1/gray/{each}/{file}'
            if not os.path.exists(f"./dog_v1/gray/{each}/"):
                os.makedirs(f"./dog_v1/gray/{each}/")
            with open(new,'w') as file:
                file.write(".")
            output.save(new,format='JPEG')
            bar()