from __future__ import print_function
import os
import numpy as np
import cv2
#from Data import image_cols, image_rows
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import pandas as pd
from pandas import DataFrame


def to_csv():
  base_path='/content/drive/MyDrive/논문/kvasir-instrument/colorview'
  images = os.listdir(base_path)
  data_df=pd.DataFrame(columns=['colorview','mask','iou','f1score'])

  i=0
  for image_name in images:
    if 'pred' in image_name:
      continue

    image_pred_name = image_name.split('.')[0]+'pred.png'

    a =cv2.imread(os.path.join(base_path,image_name))
    b=cv2.imread(os.path.join(base_path,image_pred_name))  

    cv2_imshow(b)
    #iou
    c = cv2.subtract(b, a)
    d = cv2.subtract(a, b)
    e = cv2.add(a,b)
  
    sub1 = cv2.subtract(e, c)
    z = cv2.subtract(sub1, d)

    iou_score=((np.sum(z)/255)/(np.sum(e)/255))*100

    #f1score
    precision=np.sum(z)/(np.sum(z)+np.sum(d))
    recall=np.sum(z)/(np.sum(z)+np.sum(c))
    f1score=(2*precision*recall)/(precision+recall)*100

    print(iou_score)
    print(f1score)
    data_df.loc[i]=[image_name, image_pred_name, iou_score,f1score]
    i+=1
  data_df.to_csv('/content/drive/MyDrive/논문/kvasir-instrument/colorview_re.csv',sep=',')

if __name__ == '__main__':
    to_csv()
