import pandas as pd
import cv2 
import os

# get path
directiry_cat_path = 'data/Petimages/Cat'
file_cat_path = []
directory_dog_path = 'data/PetImages/Dog'
file_dog_path = []
for file_name in os.listdir(directiry_cat_path):
    file_cat_path.append(os.path.join(directiry_cat_path, file_name))
for file_name in os.listdir(directory_dog_path):
    file_dog_path.append(os.path.join(directory_dog_path,file_name))
# init label
label_cat = [0] * len(file_cat_path)
label_dog = [1] * len(file_dog_path)
# create dataframe
cat_df = pd.DataFrame({'pixel':file_cat_path,'label':label_cat})
dog_df = pd.DataFrame({'pixel':file_dog_path,'label':label_dog})
dataframe = pd.concat([cat_df,dog_df],axis=0,ignore_index=True)
head_5000 = dataframe.head(2500)
last_5000 = dataframe.tail(2500)
dataframe = pd.concat([head_5000, last_5000], ignore_index=True)
# write dataframe to csv
dataframe.to_csv('data/label_data.csv',index=False)
print("Done")

