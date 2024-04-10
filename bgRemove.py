from rembg import remove
from PIL import Image
import os
from alive_progress import alive_bar
folds = ['french_bulldog','german_shepherd','golden_retriever','poodle','yorkshire_terrier']
for each in folds:
    print(f"working on {each} folder")
    folder_path= f"./dog_v1/{each}"
    fileLen=len([entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))])
    with alive_bar(fileLen) as bar:
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path)
            output=remove(img)
            output=output.convert('RGB')
            new = folder_path+'_r/'+file
            if not os.path.exists(folder_path+'_r/'):
                os.makedirs(folder_path+'_r/')
            with open(new,'w') as file:
                file.write(".")
            output.save(new,format='JPEG')
            bar()