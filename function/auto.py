from function.all_function import *
from cyclegan.train import run

import cv2

def get_256x256_img(img):
    #先取得68點
    ori_img_shape = shape_68(img)
    #將圖片頭像特寫且切成256x256
    img = cut_face(img,ori_img_shape)
    img_shape = shape_68(img)
    return img,img_shape

def get_hair(img):
    shape = shape_68(img)
    img = cut_face(img,shape)
    shape = shape_68(img)
    #hair,hair_mask,hair_alignment_point= extract_hair(img,shape)

    #hair = cv2.cvtColor(hair,cv2.COLOR_BGR2BGRA)

    #return hair,hair_mask,hair_alignment_point
    return  img,shape

def get_no_hair_img(img):
    shape = shape_68(img)
    nhimg,alignment_point = no_hair_img(img,shape)

    return nhimg,alignment_point,shape

def merge_img(face_img , hair_img , hair_mask , hair_pos_x , hair_pos_y , hair_width , hair_height , rotate_value):

    hair_img = cv2.cvtColor(hair_img,cv2.COLOR_BGRA2BGR)

    hair_pos_x,hair_pos_y,hair_width,hair_height = int(hair_pos_x),int(hair_pos_y),int(hair_width),int(hair_height) #str 2 int

    hair_img = cv2.resize(hair_img, (hair_width, hair_height), interpolation=cv2.INTER_AREA)
    hair_mask = cv2.resize(hair_mask, (hair_width, hair_height), interpolation=cv2.INTER_AREA)
    hair_img,hair_mask = rotate_img(hair_img,hair_mask,rotate_value)

    if hair_pos_x+hair_width-1 > 255:
        hair_width = 255 - hair_pos_x
        hair_img = hair_img[:,0:hair_width,:]
        hair_mask = hair_mask[:,0:hair_width,:]
    if hair_pos_y+hair_height-1 > 255:
        hair_height = 255 - hair_pos_y
        hair_img = hair_img[0:hair_height,:,:]
        hair_mask = hair_mask[0:hair_height,:,:]

    sub_face_img = face_img[hair_pos_y:hair_pos_y+hair_img.shape[0],hair_pos_x:hair_pos_x+hair_img.shape[1]]

    hair_mask[hair_mask!=255]=0
    hair_img = cv2.bitwise_and(hair_img,hair_img,mask=hair_mask)
    sub_face_img[hair_mask == 255] =(0,0,0)
    sub_face_img = cv2.add(sub_face_img,hair_img)
    face_img[hair_pos_y:hair_pos_y+hair_img.shape[0],hair_pos_x:hair_pos_x+hair_img.shape[1]] = sub_face_img

    cv2.imwrite("f1.jpg",face_img)
    return face_img

def change_style(img , shape):
    left_eye,right_eye,mouse = cut(img , shape)
    sketch = run("head",img)
    left_eye_sketch = run("left_eye",left_eye)
    right_eye_sketch = run("right_eye",right_eye)
    mouse_sketch = run("mouse",mouse)

    sketch = sketch.astype(np.uint8)
    left_eye_sketch = left_eye_sketch.astype(np.uint8)
    right_eye_sketch = right_eye_sketch.astype(np.uint8)
    mouse_sketch = mouse_sketch.astype(np.uint8)

    final_sketch = mix_img(shape,sketch,left_eye_sketch,right_eye_sketch,mouse_sketch)
    cv2.imwrite("f2.jpg",final_sketch)
    return final_sketch

def auto_get_hair_pos(face_img,face_shape,hair_img,hair_shape):
    #提取頭髮獲得頭髮參考位置
    hair,hair_mask,reference_vector,sub_reference_point = extract_hair(hair_img,hair_shape)

    #以臉部做為參考縮放頭髮、調整位置
    start_point,sub_reference_point,hair_size = auto_adjust(face_shape,hair,reference_vector,sub_reference_point)

    #調整頭髮大小
    hair = cv2.resize(hair, (hair_size[0],hair_size[1]), interpolation=cv2.INTER_AREA)
    hair_mask = cv2.resize(hair_mask, (hair_size[0],hair_size[1]), interpolation=cv2.INTER_AREA)

    hair_mask = cv2.inRange(hair_mask,(255),(255))
    hair = cv2.bitwise_and(hair,hair,mask = hair_mask)

    #用剛剛獲得的頭髮參數來自動調整光頭的大小，回傳光頭圖片
    seamlessclone,state = no_hair_img(face_img,face_shape,start_point,sub_reference_point)

    if state == 0:
        start_point = (start_point[0]+1,start_point[1]+2)
    else : 
        start_point = (start_point[0]+1,start_point[1]+10)
    #return 光頭圖、頭髮、起始位置(左上角)、調整後的頭髮size

    hair = cv2.cvtColor(hair,cv2.COLOR_BGR2BGRA)
    hair[hair_mask == 0] = [0,0,0,0]
    
    return seamlessclone,hair,hair_mask,start_point,hair_size