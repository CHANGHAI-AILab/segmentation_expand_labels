
import SimpleITK as sitk
import numpy as np
import cv2

from skimage.segmentation import watershed, expand_labels
from skimage.color import label2rgb
from skimage import data



import os
from glob import glob
import numpy as np
import vtk
import SimpleITK as sitk
import shutil
import scipy.ndimage as ndi
import nibabel as nib
import numpy as np
import math
from multiprocessing import  Process
from multiprocessing import Pool
import pyclesperanto_prototype as cle
from apeer_ometiff_library import io
import glob
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pyclesperanto_prototype as cle
import os
import argparse                 # argument parser
import numpy as np
import SimpleITK as sitk
import glob
from radiomics import cShape
from apeer_ometiff_library import io
import numpy as np
import pandas as pd
import shutil
import os,tarfile
def cal_jieshi_distribution(xueguan_path,tumor_path,distance_num):
    #file="/data/nnUNet/nnunet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task18_p3/labelsTs_st/01325649.nii.gz"
    src_us_vol = nib.load(xueguan_path)
    us_vol_data = src_us_vol.get_data()
    us_vol_data = (np.array(us_vol_data))
    ref_affine = src_us_vol.affine
    #print(np.unique(us_vol_data))
    print(us_vol_data.shape)
    slides2=[]
    slides3=[]
    slides4=[]
    slides5=[]
    slides6=[]
    slides7=[]
    slides8=[]
    slides9=[]
    slides10=[]
    print('us_vol_data num',np.unique(us_vol_data))
    for each in us_vol_data:
        #each[each!=1]=0
        #print(each)
        d2=each.copy()
        d3=each.copy()
        d4=each.copy()
        d5=each.copy()
        d6=each.copy()
        d7=each.copy()
        d8=each.copy()
        d9=each.copy()
        d10=each.copy()
        

        d2[d2!=2]=0
        d2[d2==2]=1

        d3[d3!=3]=0
        d3[d3==3]=1

        d4[d4!=4]=0
        d4[d4==4]=1

        d5[d5!=5]=0
        d5[d5==5]=1

        d6[d6!=6]=0
        d6[d6==6]=1

        d7[d7!=7]=0
        d7[d7==7]=1

        d8[d8!=8]=0
        d8[d8==8]=1

        d9[d9!=9]=0
        d9[d9==9]=1
        
        d10[d10!=10]=0
        d10[d10==10]=1

        slides2.append(d2)
        slides3.append(d3)
        slides4.append(d4)
        slides5.append(d5)
        slides6.append(d6)
        slides7.append(d7)
        slides8.append(d8)
        slides9.append(d9)
        slides10.append(d10)
    #evey xueguan vol

    us_vol_data2=np.array(slides2)
    us_vol_data3=np.array(slides3)
    us_vol_data4=np.array(slides4)
    us_vol_data5=np.array(slides5)
    us_vol_data6=np.array(slides6)
    us_vol_data7=np.array(slides7)
    us_vol_data8=np.array(slides8)
    us_vol_data9=np.array(slides9)
    us_vol_data10=np.array(slides10)



    """
    us_vol_data1 = us_vol_data1.transpose((2,1,0))
    us_vol_data2 = us_vol_data2.transpose((2,1,0))
    us_vol_data3 = us_vol_data3.transpose((2,1,0))
    us_vol_data4 = us_vol_data4.transpose((2,1,0))
    us_vol_data5 = us_vol_data5.transpose((2,1,0))
    us_vol_data6 = us_vol_data6.transpose((2,1,0))
    us_vol_data7 = us_vol_data7.transpose((2,1,0))
    us_vol_data8 = us_vol_data8.transpose((2,1,0))
    us_vol_data9 = us_vol_data9.transpose((2,1,0))
    """
    num=[]
    tumor_vol = nib.load(tumor_path)
    tumor_data = tumor_vol.get_data()
    tumor_data = np.array(tumor_data)
    #tumor_data = expand_labels(tumor_data,distance=1)
    tumor_data = (tumor_data)
    input_image = tumor_data


    #distance_num=8
    us_vol_data2= expand_labels(us_vol_data2,distance=distance_num)
    us_vol_data3= expand_labels(us_vol_data3,distance=distance_num)
    us_vol_data4= expand_labels(us_vol_data4,distance=distance_num)
    us_vol_data5= expand_labels(us_vol_data5,distance=distance_num)
    us_vol_data6= expand_labels(us_vol_data6,distance=distance_num)
    us_vol_data7= expand_labels(us_vol_data7,distance=distance_num)
    us_vol_data8= expand_labels(us_vol_data8,distance=distance_num)
    us_vol_data9= expand_labels(us_vol_data9,distance=distance_num)
    us_vol_data10= expand_labels(us_vol_data10,distance=distance_num)



    overlap = input_image * us_vol_data2  # Logical AND
    union = (input_image + us_vol_data2)>0  # Logical OR
    iou2 = overlap.sum() / float(union.sum())
    num.append(iou2)

    overlap = input_image * us_vol_data3  # Logical AND
    union = (input_image + us_vol_data3)>0  # Logical OR
    iou3 = overlap.sum() / float(union.sum())
    num.append(iou3)

    overlap = input_image * us_vol_data4  # Logical AND
    union = (input_image + us_vol_data4)>0  # Logical OR
    iou4 = overlap.sum() / float(union.sum())
    num.append(iou4)

    overlap = input_image * us_vol_data5  # Logical AND
    union = (input_image + us_vol_data5)>0  # Logical OR
    iou5 = overlap.sum() / float(union.sum())
    num.append(iou5)

    overlap = input_image * us_vol_data6  # Logical AND
    union = (input_image + us_vol_data6)>0  # Logical OR
    iou6 = overlap.sum() / float(union.sum())
    num.append(iou6)

    overlap = input_image * us_vol_data7  # Logical AND
    union = (input_image + us_vol_data7)>0  # Logical OR
    iou7 = overlap.sum() / float(union.sum())
    num.append(iou7)

    overlap = input_image * us_vol_data8  # Logical AND
    union = (input_image + us_vol_data8)>0  # Logical OR
    iou8 = overlap.sum() / float(union.sum())
    num.append(iou8)

    overlap = input_image * us_vol_data9  # Logical AND
    union = (input_image + us_vol_data9)>0  # Logical OR
    iou9 = overlap.sum() / float(union.sum())
    num.append(iou9)
    
    overlap = input_image * us_vol_data10  # Logical AND
    union = (input_image + us_vol_data10)>0  # Logical OR
    iou10 = overlap.sum() / float(union.sum())
    num.append(iou10)
    print(num)
    return iou2,iou3,iou4,iou5,iou6,iou7,iou8,iou9,iou10


if __name__ == '__main__':


    import pandas as pd

    distance_num=8
    
    xueguan_path_="/data/nnUNet/nnunet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task11_xueguan/xueguan.label"
    tumor_path_="/data/nnUNet/nnunet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task19_tumor/690tumormask"
    ids=[]
    nums=[]
    col10=[]
    col2=[]
    col3=[]
    col4=[]
    col5=[]
    col6=[]
    col7=[]
    col8=[]
    col9=[]
    label=[]

    for each_MASK_path in glob.glob(xueguan_path_+'/*', recursive=True):
        filename=each_MASK_path.split("/")[-1]
        xueguan_path=each_MASK_path
        #tumor_path=tumor_path_+"/"+filename
        #print(xueguan_path,tumor_path)
        if os.path.exists(tumor_path_+"/"+filename):
            pass
            #print("found")
            iou2,iou3,iou4,iou5,iou6,iou7,iou8,iou9,iou10 = cal_jieshi_distribution(xueguan_path,tumor_path_+"/"+filename)
        elif os.path.exists(tumor_path_+"/0"+filename):
            pass
            #print("found")
            iou2,iou3,iou4,iou5,iou6,iou7,iou8,iou9,iou10 = cal_jieshi_distribution(xueguan_path,tumor_path_+"/0"+filename)
        elif os.path.exists(tumor_path_+"/"+filename[1:]):
            pass
            #print("found")
            iou2,iou3,iou4,iou5,iou6,iou7,iou8,iou9,iou10 = cal_jieshi_distribution(xueguan_path,tumor_path_+"/"+filename[1:])
        else:
            print(xueguan_path)
            print("not++++++++++++++++++++++++")
        ids.append(filename)
        col10.append(iou10)
        col2.append(iou2)
        col3.append(iou3)
        col4.append(iou4)
        col5.append(iou5)
        col6.append(iou6)
        col7.append(iou7)
        col8.append(iou8)
        col9.append(iou9)
        if iou9+iou8+iou7+iou6+iou5+iou4+iou3+iou2+iou10 > 0:
            label.append('1')
            print("lable 1")
        else:
            label.append('0')
            print("lable 0")

    a = pd.DataFrame({'id':ids,'XUEGUAN10':col10,'XUEGUAN2':col2,'XUEGUAN3':col3,'XUEGUAN4':col4,'XUEGUAN5':col5,'XUEGUAN6':col6,'XUEGUAN7':col7,'XUEGUAN8':col8,'XUEGUAN9':col9,'label':label})
    a.to_csv('score_xuanguan_expand_8.csv')

    cal=[]
    data1  = pd.read_csv("/data/XIAOYU_TUMOR_CT_APV/score_tumor_expand_0.csv")
    for i in range(len(data1)):
        index_val = data1.index[i]
        row =data1.loc[index_val]
        if len(row['pid'].split(".")[0])==7:
            cal.append('0'+row['pid'])
        else:
            cal.append(row['pid'])
    print(cal)
    data1['pid']=cal
    data1.to_csv("/data/XIAOYU_TUMOR_CT_APV/score_tumor_expand_0_.csv")
