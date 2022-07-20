import os
import csv

path = '/home/bnghi/Stage_CRCT/Dataset/datasets/new_data_WSI/exemple/512x512/masks'
#path = path_to_im
files = os.listdir(path)
fields = ['input', 'label', 'set']
#f = open(r'C:\Users\nguye\Documents\Stage_CRCT\Dataset\datasets\all dataset\dataset.csv','w')
filename = "/home/bnghi/Stage_CRCT/Dataset/datasets/new_data_WSI/exemple/512x512/dataset.csv"
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
