import os
from os.path import getsize
directory = "personal_info/"
outfile_name = "labeling.txt"
os.path = "personal_info/"
out_file = open(outfile_name, 'w')

files = os.listdir(directory)

for filename in files:
    if ".txt" not in filename:
        continue
    file = open(directory + filename)

    for line in file:
        filename1 = filename[:-4]
        #.txt 제거
        filename2 = filename1+'.jpg'
        #.jpg 붙이기
        file1 = filename2
        # 이미지 파일 로드
        file_size = getsize(file1)
        # 이미지 파일 크기 가져오기

        filename3 = filename2+str(file_size)

        out_file.write(filename3)
        out_file.write('\t')
        out_file.write(str(file_size))
        out_file.write('\t')
        out_file.write(str(filename2))
        out_file.write('\t')
        out_file.write(line)
   # out_file.write("\n\n")
    file.close()
out_file.close()
