import glob
import PIL
from PIL import Image

image_list = []

for filename in glob.glob(r'C:\Users\Bernardo\Desktop\UTKFace/*.jpg'):
    im = Image.open(filename)
    image_list.append(im)

print(image_list[0])
