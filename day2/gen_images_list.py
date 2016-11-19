#-*-coding:utf-8-*-
import os
import random

def gen_images_list(data_dir='102flowers/', test_spilt =0.3):
    sub_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.abspath(os.path.join(data_dir,f)))]
    img_index = 0
    f_train = open('train.lst','w')
    f_test = open('test.lst','w')
    test_files = []
    train_files = []

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
                # f_test.write(line)
                test_files.append(line)
            else:
                train_files.append(line)

    random.shuffle(test_files)
    random.shuffle(train_files)
    for f in test_files:
        f_test.write(f)
    for f in train_files:
        f_train.write(f)
    
    f_train.close()
    f_test.close()

if __name__ == '__main__':
    gen_images_list()
