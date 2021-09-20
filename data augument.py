from PIL import Image
from PIL import ImageEnhance
import os
import numpy as np
imageDir=["/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/corrugation","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/foreign body","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/lack of fasteners","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/other defect","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/surface abrasion","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/uneven light band"]     #要改变的图片的路径文件夹
saveDir=["/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/corrugation","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/foreign body","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/lack of fasteners","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/other defect","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/surface abrasion","/Users/USER/Desktop/学校/中南大四上/毕设选题/rail images data/rail images data/uneven light band"]    #要保存的图片的路径文件夹


def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(20) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def aug_data(times):
 for count in range(times):
    print("aug data for {} times ".format(count))

    for i in range(len(imageDir)):

        for name in os.listdir(imageDir[i]):


             saveName= name[:-4]+"fl.jpg"
             saveImage=flip(imageDir[i],name)
             saveImage.save(os.path.join(imageDir[i],saveName))

             saveName= name[:-4]+"ro.jpg"
             saveImage=rotation(imageDir[i],name)
             saveImage.save(os.path.join(imageDir[i],saveName))




if __name__ == "__main__":
    times = 5
    aug_data(times)

