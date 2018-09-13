import numpy as np# -*- coding: utf-8 -*-
from PIL import Image
import os
import pickle
import argparse

parser=argparse.ArgumentParser(description='argument for generate pkl file')
parser.add_argument('--Path_Dir', type=str, default = "D:/xuexiziliao/Proj/VS PROJ/P1/RGB_IMG/")
parser.add_argument('--reshape', type=str, default='(320,240)')
parser.add_argument('--img_num_each_file', type=int, default=500)
parser.add_argument('--test_num_each_class', type=int, default=0)
args=parser.parse_args()

Path_Dir=args.Path_Dir
reshape=eval(args.reshape)
img_num_each_file=args.img_num_each_file
test_num_each_class=args.test_num_each_class




def write_log(Path_Dir):
    log_hander=open(Path_Dir+'img_log.txt','w+')
    folders=os.listdir(Path_Dir)
    index_and_name=list(enumerate(folders))
    for index_of_files,folder_name in index_and_name:
        Path_Dir_Sub=os.path.join(Path_Dir,folder_name)
        if(os.path.isdir(Path_Dir_Sub)):
            log_hander.write(Path_Dir_Sub+'\n')
            print(Path_Dir_Sub)
            img_number=len([name for name in os.listdir(Path_Dir_Sub)])
            log_hander.write('Image Nmber: {}    '.format(img_number))
            each_folder_size=0
            for name in os.listdir(Path_Dir_Sub):
                each_file_size=os.path.getsize(os.path.join(Path_Dir_Sub,name))
                each_folder_size+=each_file_size
            log_hander.write('Total Size: {:.5f} MB\n'.format((each_folder_size/1024/1024)))
    log_hander.close()

def Pickle_Convert_Img(Path_Dir,reshape=reshape,img_num_each_file=img_num_each_file,test_num_each_class=test_num_each_class): 
    folders=os.listdir(Path_Dir)
    index_and_name=list(enumerate(folders))
    whole_num=0
    folder_num=0
    for index_of_files,folder_name in index_and_name:
        Path_Dir_Sub=os.path.join(Path_Dir,folder_name)
        if(os.path.isdir(Path_Dir_Sub)):
            folder_num+=1
            img_number=len([name for name in os.listdir(Path_Dir_Sub)])
            whole_num+=img_number
    Test_Whole_Num=test_num_each_class*folder_num
    Train_Whole_Num=whole_num-Test_Whole_Num
    Train_img_array_total=np.array([])
    Train_img_label_total=np.array([])
    Test_img_array_total=np.array([])
    Test_img_label_total=np.array([])
    
    Train_img_current_index=0#current stored img number
    Test_img_current_index=0
    Train_File_Number=0
    Test_File_Number=0
    Trian_Pointer=0
    Test_Pointer=0
    
    
    for index_of_files,folder_name in index_and_name:
        Path_Dir_Sub=os.path.join(Path_Dir,folder_name)
        if(os.path.isdir(Path_Dir_Sub)):
            pic_name=[i for i in os.listdir(Path_Dir_Sub)]
            label=int(folder_name[2:])*np.ones((1,1))
            pic_name_num=np.size(pic_name)
            for pic_index in range(pic_name_num):
                Single_Img=Image.open(os.path.join(Path_Dir_Sub,pic_name[pic_index]))
                print('{},{}'.format(os.path.join(Path_Dir_Sub,pic_name[pic_index]),pic_index))
                print(reshape)
                Single_Img=Single_Img.resize(reshape)
                img_array=np.asarray(Single_Img)
                img_shape=np.shape(img_array)
                print(np.shape(img_shape))
                if(np.size(img_shape)==2): #gray img
                    img_array=np.expand_dims(img_array,axis=0)
                    img_array_temp=img_array
                    for i in range(2):
                        img_array_temp=np.append(img_array_temp,img_array,axis=2)
                    img_array=img_array_temp
                    img_shape=np.shape(img_array)
                    print(np.shape(img_shape))
                if(np.size(img_shape)==3 and img_shape[2]==4 ): #have4 channel
                    img_array=img_array[:,:,:3]
                    img_shape=np.shape(img_array)          
                img_array=np.reshape(img_array,(1,img_shape[0]*img_shape[1]*img_shape[2]))
                print(np.shape(img_array))
                
                train_num_per_folder=pic_name_num-test_num_each_class
                try:
                    if(train_num_per_folder<0):
                        raise Exception('TestNumIllegal')
                except Exception as e:
                    print(e)
                    print('The test batch number is larger than total number: total num:{}, test num:{}'.format(pic_name_num,test_num_each_class))
                    break
                
                if(pic_index<train_num_per_folder):#deal with train
                    print('total_train_num.............:{}'.format(train_num_per_folder))
                    if(Train_img_array_total.size==0 and Train_img_label_total.size==0):
                        Train_img_array_total=img_array#
                        Train_img_label_total=label
                        print('init train size: {}'.format(Train_img_array_total.size))
                    else:
                        print('concatenate train')
                        print(img_array.shape)
                        Train_img_array_total=np.concatenate((Train_img_array_total,img_array),axis=0)
                        Train_img_label_total=np.concatenate((Train_img_label_total,label),axis=1)
                
                    Train_img_current_index+=1
                    print('total num: {}, total train num:{},current index:{}'.format(whole_num,Train_Whole_Num,Train_img_current_index))
                    print('current size:{}'.format(np.size(Train_img_label_total)))
                    if(np.size(Train_img_label_total)>=img_num_each_file or (Train_Whole_Num%img_num_each_file!=0 and Train_img_current_index==Train_Whole_Num)):
                        img_data={'image':Train_img_array_total}
                        img_label={'label':Train_img_label_total}
                        Train_File=open(os.path.join(Path_Dir,'Train_Data{}.pkl'.format(Train_File_Number)),'wb')
                        
                        pickle.dump(img_data,Train_File,protocol=pickle.HIGHEST_PROTOCOL)#binary file
                        pickle.dump(img_label,Train_File,protocol=pickle.HIGHEST_PROTOCOL)#binary file
                        Train_File.close
                        Train_File_Number+=1
                        print('Train file_number:{}'.format(Train_File_Number))
                        Train_img_array_total=np.array([])
                        Train_img_label_total=np.array([])
                        
                if(pic_index>=train_num_per_folder):#deal with test
                    print('total_test_num.............:{}'.format(train_num_per_folder))
                    if(Test_img_array_total.size==0 and Test_img_label_total.size==0):
                        Test_img_array_total=img_array#
                        Test_img_label_total=label
                        print('init test size: {}'.format(Test_img_array_total.size))
                    else:
                        print('concatenate test')
                        print(img_array.shape)
                        Test_img_array_total=np.concatenate((Test_img_array_total,img_array),axis=0)
                        Test_img_label_total=np.concatenate((Test_img_label_total,label),axis=1)
                    Test_img_current_index+=1
                    print('total num: {}, total test num:{},current index:{}'.format(whole_num,Test_Whole_Num,Test_img_current_index))
                    print('current size:{}'.format(np.size(Test_img_label_total)))
                    if(np.size(Test_img_label_total)>=img_num_each_file or (Test_Whole_Num%img_num_each_file!=0 and Test_img_current_index==Test_Whole_Num)):
                        img_data={'image':Test_img_array_total}
                        img_label={'label':Test_img_label_total}
                        Test_File=open(os.path.join(Path_Dir,'Test_Data{}.pkl'.format(Test_File_Number)),'wb')
                        pickle.dump(img_data,Test_File,protocol=pickle.HIGHEST_PROTOCOL)#binary file
                        pickle.dump(img_label,Test_File,protocol=pickle.HIGHEST_PROTOCOL)#binary file
                        Test_File.close
                        Test_File_Number+=1
                        print('Test file number:{}'.format(Test_File_Number))
                        Test_img_array_total=np.array([])
                        Test_img_label_total=np.array([])
def load_pickle():
    testfile2=open(os.path.join(Path_Dir,'Test_Data0.pkl'),'rb')
    img=pickle.load(testfile2)
    label=pickle.load(testfile2)
    testfile2.close
    print('for test')




if __name__=='__main__':
    write_log(Path_Dir)
    #convert_image_to_CSV(Path_Dir)
    #Pickle_Convert_Img(Path_Dir)
    load_pickle()
