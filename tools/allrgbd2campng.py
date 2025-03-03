#把所有rgbd目录下的rgb.npy图片转换到对应cam目录下的png图片
import os
import cv2
import numpy as np

# 设置输出视频为mp4格式
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

# cap_fps是帧率，可以根据随意设置
cap_fps = 10

# 注意！！！
# size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），但是此处的size要传的是（width，height），这里一定要注意注意不然结果会打不开，比如通过img.shape得到常用的图片尺寸
#size = (1110, 1013)

# 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
# video = cv2.VideoWriter(r'D:\video.mp4',fourcc, cap_fps, size, isColor=0)
#video = cv2.VideoWriter(r'task_record.mp4', fourcc, cap_fps, size)

# 这里直接读取目录下的所有图片。   
doc=0
en = 1
while True:
  edir = 'episode-'+str(en)
  if not os.path.exists(edir):
    break
  rn=1
  while True:
    rdir = edir+'/rgbd-'+str(rn)
    if not os.path.exists(rdir):
      break
    cdir = edir+'/cam-'+str(rn)
    
    
    path = rdir+'/'
    file_lst = os.listdir(path)
    file_lst.sort()
    for filename in file_lst:
        if filename[-5:] != 'b.npy':
          continue
        f1 = path + filename
        data = np.load(f1)
        data = np.flip(data,axis=0)
        img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        filetitle = filename[:-8]
        f2 = cdir + '/' + filetitle + '.png'
        cv2.imwrite(f2,img)
        doc += 1
        print(f'[{doc}] {f1} => {f2}')
    
    
    rn += 1
  en += 1

