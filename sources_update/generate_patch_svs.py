from email import parser
from pathaia.util.types import Slide,Patch
import  matplotlib.pyplot as plt
from PIL import Image
from pathaia.patches.functional_api import slide_rois
from pathaia.patches import filter_thumbnail
import numpy as np
import argparse
import os
import csv


# parser = argparse.ArgumentParser(description='create dataset of 512x512 image from WSI')
# parser.add_argument("WSI_path", help="enter the path of WSI file")
# args = parser.parse_args()

slide = Slide("/home/bnghi/Stage_CRCT/Dataset/datasets/microcal/WSI/21I000004-1-03-4_141003.svs")

patches = slide_rois(
        slide,
        level=0,
        psize=512,
        interval=0,
        slide_filters=[filter_thumbnail],
        thumb_size=2000,
    )

a = "21I000004-1-03-4_141003"

for patch in patches:
    im = Image.fromarray(patch[1])
    x = str(patch[0].position[0])
    y = str(patch[0].position[1])
    im.save("../../../Dataset/datasets/microcal/512_from_WSI/" + a + "_" + x + "_" + y + ".png")

path = '/home/bnghi/Stage_CRCT/Dataset/datasets/microcal/512_from_WSI'
#path = path_to_im
files = os.listdir(path)
fields = ['input', 'label', 'set']
#f = open(r'C:\Users\nguye\Documents\Stage_CRCT\Dataset\datasets\all dataset\dataset.csv','w')
filename = "../../../Dataset/datasets/microcal/1024x1024/dataset.csv"
#filename = path_to_csv
k = []
for file in files:
    p = ['images/'+file, 'masks/'+file, 'testing']
    k.append(p)

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile, lineterminator='\n') 
        
    # writing the fields 
    csvwriter.writerow(fields) 
    
    #f.writerows("images" + filename)
    # writing the data rows 
    csvwriter.writerows(k)

csvfile.close()
