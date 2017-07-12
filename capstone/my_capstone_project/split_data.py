import numpy as np
from PIL import Image, ImageDraw
import os
import random
import shutil

train_path = 'Train/'
dotted_train_path = 'TrainDotted/'

validate_train_path = 'Test/'
if not os.path.exists(validate_train_path):
    os.makedirs(validate_train_path)

validate_dotted_train_path = 'TestDotted/'
if not os.path.exists(validate_dotted_train_path):
    os.makedirs(validate_dotted_train_path)

i = 0
while i < 100:
    img_num = random.randint(1, 948)
    img_name = str(img_num) + '.jpg'
    if os.path.isfile(train_path+img_name) == False:
        continue
    else:
        src_train = train_path+img_name
        dest_train = validate_train_path + img_name
        shutil.move(src_train, dest_train)

        src_dotted = dotted_train_path + img_name
        dest_dotted = validate_dotted_train_path + img_name
        shutil.move(src_dotted, dest_dotted)

        i += 1
    print i

print i
print ("Done")



