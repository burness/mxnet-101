#-*-coding:utf-8-*-
import os
import random

def gen_images_list(data_dir='17flowers/', test_spilt =0.3):
    sub_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.abspath(os.path.join(data_dir,f)))]
    img_index = 0
    f_train = open('train.lst','w')
    f_test = open('test.lst','w')

    for sub_dir in sub_dirs:
        c_dir = os.path.join(data_dir, sub_dir)
        img_label = sub_dir
        c_dir_files = [os.path.join(sub_dir,f) for f in os.listdir(c_dir) if os.path.isfile(os.path.abspath(os.path.join(c_dir,f)))]
        c_dir_files_len = len(c_dir_files)
        c_dir_files_test_len = int(test_spilt*c_dir_files_len)
        random.shuffle(c_dir_files)
        for index,file in enumerate(c_dir_files):
            line = str(img_index)+'\t'+img_label+'\t'+file+'\n'
            img_index+=1
            if index < c_dir_files_test_len:
                f_test.write(line)
            else:
                f_train.write(line)
    f_train.close()
    f_test.close()

if __name__ == '__main__':
    gen_images_list()
