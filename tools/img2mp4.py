#图片转视频
import os
import cv2

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
path = 'episode-1/cam-1/'
file_lst = os.listdir(path)
file_lst.sort()
doc=0
for filename in file_lst:
    img = cv2.imread(path + filename)
    if doc==0:
      h, w = img.shape[:2]
      print(f'img.size={w,h}')
      video = cv2.VideoWriter(r'task_record.mp4', fourcc, cap_fps, (w,h))
    #图片可以裁剪
    #img = img[:, :]
    #图片可以resize
    # img = cv2.resize(img, (960, 470), interpolation=cv2.INTER_AREA)
    #输出前可视化查看
    # cv2.imshow("output", img)
    # cv2.waitKey(0)
    video.write(img)
    doc += 1

