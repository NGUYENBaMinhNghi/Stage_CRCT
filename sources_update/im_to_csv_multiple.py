import os
import csv
import numpy as np

path = '../../Dataset/datasets/microcal/512x512_1/images'
#path = path_to_im
files = os.listdir(path)
fields = ['input', 'label', 'set']
#f = open(r'C:\Users\nguye\Documents\Stage_CRCT\Dataset\datasets\all dataset\dataset.csv','w')
filename = "../../Dataset/datasets/microcal/512x512_1/dataset.csv"
#filename = path_to_csv

B = np.arange(0, len(files))
liste_num = np.random.choice(B, 15)

k = []
for i in liste_num:
    p = ['images/'+files[i], 'masks/'+files[i], 'training']
    k.append(p)


# writing to csv file 
with open(filename, 'w') as csvfile: 
    csvfile.truncate()

    # creating a csv writer object 
    csvwriter = csv.writer(csvfile, lineterminator='\n') 
        
    # writing the fields 
    csvwriter.writerow(fields) 
    
    #f.writerows("images" + filename)
    # writing the data rows 
    csvwriter.writerows(k)

csvfile.close()
