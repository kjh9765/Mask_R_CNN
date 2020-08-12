import json
import os

tmpArr = ''
with open('labeling.txt','r') as file:
    for text in file:
        array=text.split()

        String='\"'+array[0]+'\":'+'{'+"\"fileref\":\"\",\"size\":"+array[1]+",\"filename\":\""+array[2]+'\",\"base64_img_data\":\"\",\"file_attributes\":{},\"regions\":{\"0\":{\"shape_attributes\":{\"name\":\"polygon\",\"all_points_x\":['+array[3]+','+str(int((int(array[5])+int(array[3]))/2))+','+ array[5]+','+array[5]+','+array[5]+','+str(int((int(array[5])+int(array[3]))/2))+','+array[3]+','+array[3]+'],\"all_points_y\":['+array[4]+','+array[4]+','+array[4]+','+str(int((int(array[6])+int(array[4]))/2))+','+array[6]+','+array[6]+','+array[6]+','+str(int((int(array[6])+int(array[4]))/2))+']},\"region_attributes\":{\"object\":\"'+array[7]+'\"}}}},'
        tmpArr = tmpArr+String

jsonArr = '{'+tmpArr[:-1]+'}'


directory = "personal_info/"
outfile_name = "via_region_data.json"
out_file = open(outfile_name, 'w')

out_file.write(jsonArr)
file.close()
out_file.close()




