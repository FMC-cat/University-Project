import cv2,dlib
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from imutils import face_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
'''小工具'''

#將x,y換成index
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

#取得68點
def shape_68(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("function/shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
    return shape

#把img轉換成uv_img
def get_uv(img):
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    u_channel = luv[:, :, 1].astype(np.float32)
    v_channel = luv[:, :, 2].astype(np.float32)

    uv_img = np.sqrt(np.square(u_channel) + np.square(v_channel))
    uv_img = cv2.normalize(uv_img, None, 0, 255, cv2.NORM_MINMAX)

    #把背景設定成-900之後才不會對分群造成問題
    bg_mask = cv2.inRange(img,(255,255,255),(255,255,255))
    uv_img[bg_mask==255] = -900
    return uv_img

#把uv_img分群
def kmean_uv(img_uv):
    cluster_means = [74.9965790361166, 215.2911108656476, 123.2515908241272, 38.05951690673828,-900]
    n_clusters = len(cluster_means)
    init_centers = np.array(cluster_means).reshape((-1, 1))

    # 使用reshape轉換圖像陣列為1D陣列
    data = img_uv.reshape((-1, 1))

    # 使用KMeans演算法進行聚類
    kmeans = KMeans(n_clusters=n_clusters, init=init_centers, max_iter=100).fit(data)

    # 得到聚類結果
    clusters = kmeans.predict(data)

    # 將聚類結果轉換回圖像形式
    img_clustered = clusters.reshape((img_uv.shape))

    return img_clustered

#把區域A和C用線性插值接在一起
def interpolate(area_A,area_C):
    area_A = cv2.cvtColor(area_A,cv2.COLOR_BGR2LUV)
    area_C = cv2.cvtColor(area_C,cv2.COLOR_BGR2LUV)
    #取得最後一欄
    column1 = area_A[:, -1, :]
    column1 = np.expand_dims(column1, axis=1)
    #取得第一欄
    column2 = area_C[:, 0, :]
    column2 = np.expand_dims(column2, axis=1)

    #開始線性差值
    num_interpolations = 10

    area_AC = np.zeros((area_A.shape[0], area_A.shape[1] + area_C.shape[1] + num_interpolations, area_A.shape[2]), dtype=area_A.dtype)
    area_AC[:, :area_A.shape[1], :] = area_A

    for i in range(1, num_interpolations + 1):
        weight = i / (num_interpolations + 1)
        interpolated_image = (1 - weight) * column1 + weight * column2
        area_AC[:, area_A.shape[1]-1 + i, :] = np.squeeze(interpolated_image, axis=1)

    area_AC[:, area_A.shape[1]+num_interpolations:, :] = area_C
    area_AC = cv2.cvtColor(area_AC,cv2.COLOR_LUV2BGR)
    return area_AC
#-----------------------------------------------------------

'''主要功能'''
def cut_face(img,shape):
    #將圖片切割成256,256

    x0,y0 = shape[27][0]-2*(shape[27][0]-shape[0][0]) , shape[27][1]-2*(shape[27][0]-shape[0][0])
    x1,y1 = shape[27][0]+2*(shape[27][0]-shape[0][0]) , shape[27][1]+2*(shape[27][0]-shape[0][0])
    X0,Y0 = x0,y0
    X1,Y1 = x1,y1
    if x0 < 0 :
        X0 = 0
    if x1 >= img.shape[1] :
        X1 = img.shape[1]-1
    if y0 < 0 :
        Y0 = 0
    if y1 >= img.shape[0] :
        Y1 = img.shape[0]-1
    img = cv2.copyMakeBorder(img[Y0:Y1,X0:X1,:],Y0-y0,y1-Y1,X0-x0,x1-X1,cv2.BORDER_CONSTANT,value=[254,254,254])
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    segmentor = SelfiSegmentation()
    img = segmentor.removeBG(img, (255,255,255), threshold=0.99)

    return(img)

#新版刪除頭髮
def delete_hair(img,shape):
    uv_img = get_uv(img)
    cluster = kmean_uv(uv_img)

    #定義顏色[3,0,2,1]
    #color_mean = [3,0,2,1]

    #臉部mask
    hull = cv2.convexHull(shape)
    face_mask = np.zeros((256,256,1),np.uint8)
    cv2.fillPoly(face_mask,[hull],(255))

    #第一群
    cluster1_mask = np.zeros((256,256,1),np.uint8)
    cluster1_mask[cluster == 3] = 255

    #第二群
    cluster2_mask = np.zeros((256,256,1),np.uint8)
    cluster2_mask[cluster == 0] = 255

    #第三群
    cluster3_mask = np.zeros((256,256,1),np.uint8)
    cluster3_mask[cluster == 2] = 255

    #第四群
    cluster4_mask = np.zeros((256,256,1),np.uint8)
    cluster4_mask[cluster == 1] = 255

    #第一群數量
    sub_clister1_mask = cluster1_mask[0:shape[30][1],:]
    clister1_pixel = np.sum(sub_clister1_mask==255)

    #第二群數量
    sub_clister2_mask = cluster2_mask[0:shape[19][1]-5,:]
    clister2_pixel = np.sum(sub_clister2_mask==255)

    #第三群數量
    sub_clister3_mask = cluster3_mask[0:shape[19][1]-5,:]
    clister3_pixel = np.sum(sub_clister3_mask==255)

    """判斷頭髮情況主要第一群、主要第一群一些第二群、主要第二群、主要第二群一些第三群、主要第三群"""
    hair_mask = np.zeros((256,256,1),np.uint8)
    face_mask_kmean = np.zeros((256,256,1),np.uint8)

    if clister1_pixel > 200:
        # print("主要第一群")
        cluster2_inface_mask = cv2.bitwise_and(cluster2_mask,cluster2_mask,mask = face_mask)
        cluster2_not_inface_mask = cv2.bitwise_and(cluster2_mask,cluster2_mask,mask = cv2.bitwise_not(face_mask))
        cluster2_not_inface_mask = cluster2_not_inface_mask[0:shape[57][1],:]   #去掉衣服

        cluster2_inface_pixel = np.sum(cluster2_inface_mask == 255)
        cluster2_not_inface_pixel = np.sum(cluster2_not_inface_mask == 255)
        if(cluster2_not_inface_pixel > cluster2_inface_pixel):
            # print("一些第二群")
            hair_mask = cv2.add(cluster1_mask,cluster2_mask)
            face_mask_kmean = cv2.add(cluster3_mask,cluster4_mask)
        else:
            hair_mask = cluster1_mask
            face_mask_kmean = cv2.add(cv2.add(cluster2_mask,cluster3_mask),cluster4_mask)
    elif clister2_pixel > 200:
        # print("主要第二群")
        cluster3_inface_mask = cv2.bitwise_and(cluster3_mask,cluster3_mask,mask = face_mask)
        cluster3_not_inface_mask = cv2.bitwise_and(cluster3_mask,cluster3_mask,mask = cv2.bitwise_not(face_mask))
        cluster3_not_inface_mask = cluster3_not_inface_mask[0:shape[57][1],:]   #去掉衣服

        cluster3_inface_pixel = np.sum(cluster3_inface_mask == 255)
        cluster3_not_inface_pixel = np.sum(cluster3_not_inface_mask == 255)
        if(cluster3_not_inface_pixel>cluster3_inface_pixel):
            # print("一些第三群")
            hair_mask = cv2.add(cluster1_mask,cluster2_mask)
            hair_mask = cv2.add(hair_mask,cluster3_mask)
            face_mask_kmean = cluster4_mask
        else:
            hair_mask = cv2.add(cluster1_mask,cluster2_mask)
            face_mask_kmean = cv2.add(cluster3_mask,cluster4_mask)
    elif clister3_pixel > 200:
        # print("主要第三群")
        hair_mask = cv2.add(cluster1_mask,cluster2_mask)
        hair_mask = cv2.add(hair_mask,cluster3_mask)
        face_mask_kmean = cluster4_mask

    img_show = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hair_mask_show = cv2.cvtColor(hair_mask,cv2.COLOR_BGR2RGB)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    hair_mask = cv2.dilate(hair_mask, kernel)
    hair_mask = cv2.erode(hair_mask, kernel)

    cluster1_mask_f = cv2.bitwise_and(face_mask,cluster1_mask)
    cluster2_mask_f = cv2.bitwise_and(face_mask,cluster2_mask)
    cluster3_mask_f = cv2.bitwise_and(face_mask,cluster3_mask)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    # axes[0, 0].imshow(cluster1_mask)
    # axes[0, 0].set_title("cluster1_mask")

    contours , hierarchy = cv2.findContours(hair_mask,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        if area<200 or y>shape[19][1]:
            hair_mask[y:y+h,x:x+w] = (0)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # hair_mask = cv2.dilate(hair_mask, kernel)
    # face_mask_not = cv2.bitwise_not(face_mask)
    # hair_mask = cv2.bitwise_and(face_mask_not,hair_mask)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # hair_mask = cv2.erode(hair_mask, kernel)
    # hair_mask = cv2.dilate(hair_mask, kernel)

    # axes[0, 1].imshow(hair_mask)
    # axes[0, 1].set_title("hair_mask")

    #檢查眉毛點有沒有在hair_mask中
    #左邊檢測
    left_eyebrow_count = 0
    for i in range(17,22):
        if hair_mask[shape[i][1],shape[i][0]] == 255:

            left_eyebrow_count += 1
    #右邊檢測
    right_eyebrow_count = 0
    for i in range(22,27):
        if hair_mask[shape[i][1],shape[i][0]] == 255:

            right_eyebrow_count += 1
    clone = img.copy()
    forehead_points = []
    if(left_eyebrow_count >= 2 or right_eyebrow_count >= 2):
        #拿到臉部輪廓
        contours,_ = cv2.findContours(face_mask_kmean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                contour_points = contour.reshape(-1, 2)
                for points in contour_points:
                    if (points[0] > shape[17][0] and points[0] < shape[26][0]) and (points[1] < shape[29][1]):
                        forehead_points.append((points[0],points[1]))
    #計算髮線點高於眉毛的數量
    forehead_count = 0
    #0是眉毛沒有被遮住,1是眉毛被遮住了
    state = 0
    if len(forehead_points) > 0:
        state = 1
        for x in range(shape[17][0]+1,shape[26][0]):
            filtered_points = [point for point in forehead_points if point[0] == x]
            if filtered_points:
                min_y_point = min(filtered_points, key=lambda point: point[1])
                if min_y_point[1] < shape[24][1]:
                    forehead_count += 1

    #高於眉毛的數量多於兩眼之間的一半判斷為遮罩錯誤
    if forehead_count > (shape[26][0]-shape[17][0])//2:
        state = 0
        eyebrow_mask = np.zeros((256,256,1),np.uint8)
        #左眼遮住
        if left_eyebrow_count >= 2:
            #eyebrow_mask = np.zeros((256,256,1))
            left_eyebrow_indices = list(range(17, 22))
            left_eyebrow_points = np.array([(shape[idx][0],shape[idx][1]) for idx in left_eyebrow_indices])
            cv2.fillPoly(eyebrow_mask, [left_eyebrow_points], 255)

        #右眼遮住
        if right_eyebrow_count >= 2:
            right_eyebrow_indices = list(range(22, 27))
            right_eyebrow_points = np.array([(shape[idx][0],shape[idx][1]) for idx in right_eyebrow_indices])
            cv2.fillPoly(eyebrow_mask, [right_eyebrow_points], 255)

        #再對頭髮做一次bitwise_and
        hair_mask = cv2.bitwise_and(hair_mask,hair_mask,mask = cv2.bitwise_not(eyebrow_mask))

        #補上眉毛後再次替除小點
        contours , hierarchy = cv2.findContours(hair_mask,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            if area<500 or y>shape[19][1]:
                hair_mask[y:y+h,x:x+w] = (0)

    face_img = img.copy()
    face_img[hair_mask == 255] = (255,255,255)

    # fig.suptitle("")
    # plt.tight_layout()
    # plt.show()

    return face_img,hair_mask,face_mask_kmean,state

def no_hair_img(img,shape,start_point,sub_reference_point):
    face_img,hair_mask,_,state = delete_hair(img,shape)

    face_mask = cv2.bitwise_not(cv2.inRange(face_img,(255,255,255),(255,255,255)))

    #橢圓中心
    center_x = start_point[0]+((sub_reference_point[0][0]+sub_reference_point[1][0]) // 2)+1
    if(state == 0):
        center_y = start_point[1]+((sub_reference_point[0][1]+sub_reference_point[1][1]) // 2) - 3
    else : #沒有眉毛
        center_y =  start_point[1]+((sub_reference_point[0][1]+sub_reference_point[1][1]) // 2) + 10


    #橢圓的寬和高
    head_width = abs(sub_reference_point[0][0]-sub_reference_point[1][0])+2
    head_height = abs(center_y-(start_point[1]+(sub_reference_point[2][1]//2)))
    #head_height = abs(center_y-(start_point[1]+(sub_reference_point[2][1])))
    ellipse_mask = np.zeros((256, 256, 1), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, (center_x, center_y), (int(head_width / 2), int(head_height)), 0, 165, 375, 255, -1)
    ellipse_mask[shape[28][1]+5:255,:] = 0


    #把face_mask變小(侵蝕)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    small_face_mask = cv2.erode(face_mask,kernel)

    forehead_mask = cv2.bitwise_and(ellipse_mask,ellipse_mask,mask = small_face_mask)

    #取顏色參考點
    contours, _ = cv2.findContours(forehead_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    merged_contour = np.concatenate(contours) #合併輪廓(有可能是分開的)

    #紀錄額頭點的最小y值
    min_y_values = {}
    for point in merged_contour:
        x, y = point[0]
        if x not in min_y_values or y < min_y_values[x]:
            min_y_values[x] = y


    skin_color_patch = np.zeros_like(face_img)   #皮膚補丁

    #取得頭髮的最大和最小y
    contours , hierarchy = cv2.findContours(hair_mask,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

    min_y = 0
    max_y = face_img.shape[0]

    #眉毛mask，如果取色點在裡面直接用前一個點
    left_eyebrow = shape[17:22]
    right_eyebrow = shape[22:27]
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    eye_eyebrow_mask = np.zeros((256,256)).astype(np.uint8)

    cv2.fillPoly(eye_eyebrow_mask, [left_eyebrow], 255)
    cv2.fillPoly(eye_eyebrow_mask, [right_eyebrow], 255)
    cv2.fillPoly(eye_eyebrow_mask, [left_eye], 255)
    cv2.fillPoly(eye_eyebrow_mask, [right_eye], 255)
    # cv2.imshow("eye_eyebrow_mask",eye_eyebrow_mask)
    # cv2.waitKey()

    #填充皮膚補丁
    skin_color_patch_left = center_x - int(head_width / 2)
    skin_color_patch_right = center_x + int(head_width / 2)

    #若不在字典裡用前一個的顏色
    pre_reference = min(min_y_values.keys())

    for x in range(skin_color_patch_left,skin_color_patch_right+1):
        if x not in min_y_values or (eye_eyebrow_mask[min_y_values[x]][x] == 255):
            skin_color_patch[min_y:max_y,x] = img[min_y_values[pre_reference]][pre_reference]
        else:
            skin_color_patch[min_y:max_y,x] = img[min_y_values[x]][x]
            pre_reference = x

    #模糊處理
    skin_color_patch  = cv2.medianBlur(skin_color_patch, 5)

    skin_color_patch_mask = cv2.bitwise_and(ellipse_mask,ellipse_mask,mask = cv2.bitwise_not(forehead_mask))
    #skin_color_patch_mask[shape[29][1]:255,:] = 0

    face_img[skin_color_patch_mask[:,:] == 255] = (0,0,0)
    skin_color_patch = cv2.bitwise_and(skin_color_patch,skin_color_patch,mask = skin_color_patch_mask)

    nhimg = cv2.add(face_img,skin_color_patch)

    #柏松融合
    forehead_mask = cv2.bitwise_and(ellipse_mask,ellipse_mask,mask = cv2.bitwise_not(forehead_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    seamlessclone_mask = cv2.dilate(forehead_mask, kernel)

    x, y, w, h = cv2.boundingRect(seamlessclone_mask)
    sub_seamlessclone_mask = seamlessclone_mask[y:y+h,x:x+w]

    #A和C區域分別為臉頰左側和右側
    area_A = img[shape[28][1]:shape[33][1],shape[3][0]:shape[60][0]]
    area_C = img[shape[28][1]:shape[33][1],shape[64][0]:shape[13][0]]
    area_AC = interpolate(area_A,area_C)
    area_AC = cv2.resize(area_AC, (w, h), interpolation=cv2.INTER_AREA)
    area_AC.astype(np.uint8)

    sub_seamlessclone_mask.astype(np.uint8)
    seamlessclone = cv2.seamlessClone(area_AC, nhimg, sub_seamlessclone_mask, (x+(w//2),y+(h//2)), cv2.NORMAL_CLONE)

    bg_mask = cv2.inRange(nhimg,(255,255,254),(255,255,255))     #柏松融合會融合到一些背景(把背景還原)
    seamlessclone[bg_mask[:,:] == 255] = (255,255,255)
    return seamlessclone,state

def add_hair_bangs(hair,hair_mask,hair_2,image_hair,hair_shape):

    def hair_point(hair_mask):
        hair_edges = cv2.Canny(hair_mask, 30, 150)
        contours,_ = cv2.findContours(hair_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hair_edges_points = []
        for contour in contours:
            for point in contour:
                hair_edges_points.append(point[0])

        return hair_edges_points
    # 移除多餘的點 和 增加眉毛點
    def del_unimportant_point(hair_mask_points,mask):
        num_points = len(hair_mask_points)
        center_x = sum(point[0] for point in hair_mask_points) / num_points
        center_y = sum(point[1] for point in hair_mask_points) / num_points

        for point in hair_mask_points:
            if center_x - 30 <= point[0] <= center_x + 30:
                if point[1] <= center_y:
                    x, y = point
                    center_y -= 1
                else:
                    break
        center_point = (center_x,center_y)

        # for point in hair_mask_points:
        #     cv2.circle(mask, (int(point[0]),int(point[1])), 2, 255, -1)

        # cv2.circle(mask, (int(center_x),int(center_y)), 2, 100, -1)

        eyebrow_points = []
        for i in range(17,26+1):
            x,y = hair_shape[i]
            y-=3
            eyebrow_points.append((x,y))
        eyebrow_points = eyebrow_points[::-1]

        left_lowest_point = eyebrow_points[9]
        right_lowest_point = eyebrow_points[0]

        hair_mask_points = [point for point in hair_mask_points
                    if (left_lowest_point[0] <= point[0] <= right_lowest_point[0])
                    and (point[1]>=center_point[1])]
        hair_mask_points = sorted(hair_mask_points, key=lambda p: p[0])

        # for point in hair_mask_points:
        #     cv2.circle(mask, (int(point[0]),int(point[1])), 2, 255, -1)
        # cv2.imshow("mask",mask)
        # cv2.waitKey()

        hair_mask_points.extend(eyebrow_points)

        # for point in eyebrow_points:
        #     cv2.circle(mask, (int(point[0]),int(point[1])), 2, 255, -1)
        # cv2.imshow("mask",mask)
        # cv2.waitKey()

        return hair_mask_points
    # 移除相近的膚色
    def remove_similar_color_pixels(hair_shape,image_hair, hairline_to_eyebrow_mask, tolerance=70):
        # 取得目標像素，取出27點
        target_point = (hair_shape[28][0], hair_shape[28][1])
        # 取出像素顏色值
        target_color = image_hair[target_point[1], target_point[0]]
        # print("target_color",target_color)
        # 計算上下限
        lower_bound = np.array([max(0, c - tolerance) for c in target_color])
        # print("lower_bound",lower_bound)

        upper_bound = np.array([min(255, c + tolerance) for c in target_color])
        # print("lower_bound",lower_bound)

        mask = cv2.inRange(hairline_to_eyebrow_mask, lower_bound, upper_bound)

        img = hairline_to_eyebrow_mask.copy()

        img[mask > 0] = [0, 0, 0]
        bangs_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bangs_mask[bangs_mask > 0] = 255

        return img,bangs_mask

    width,height = hair_mask.shape[:2]
    mask = np.zeros((height,width), dtype=np.uint8)
    # 取得頭髮遮罩的邊緣點
    hair_mask_points = hair_point(hair_mask)
    # 移除多餘頭髮遮罩點只保留瀏海的部份，並且增加眉毛點
    hair_mask_points = del_unimportant_point(hair_mask_points,mask.copy())
    # 建立瀏海與眉毛之間的遮罩
    hair_bangs_mask = mask.copy()
    hair_mask_points_np = np.array(hair_mask_points)
    cv2.fillPoly(hair_bangs_mask, [hair_mask_points_np], (255, 255, 255))
    # cv2.imshow("hair_bangs_mask",hair_bangs_mask)
    # cv2.waitKey()
    hair_bangs_mask = cv2.bitwise_and(image_hair,image_hair,mask = hair_bangs_mask)
    # cv2.imshow("hair_bangs_mask",hair_bangs_mask)
    # cv2.waitKey()
    # 移除相近的膚色
    bangs, hair_bangs_mask = remove_similar_color_pixels(hair_shape,image_hair,hair_bangs_mask.copy())

    # cv2.imshow("hair_bangs_mask",hair_bangs_mask)
    # cv2.waitKey()
    # cv2.imshow("bangs",bangs)
    # cv2.waitKey()
    # 建立完整頭髮遮罩
    result_hair_mask = cv2.add(hair_mask,hair_bangs_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result_hair_mask = cv2.erode(result_hair_mask, kernel)
    result_hair = cv2.bitwise_and(image_hair,image_hair,mask = result_hair_mask)

    # 建立完整頭髮
    # result_hair = cv2.bitwise_or(hair,bangs)

    # cv2.imshow("hair",hair)
    # cv2.waitKey()

    return result_hair,result_hair_mask

def extract_hair(img,shape):
    _,hair_mask,_,_ = delete_hair(img,shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    hair_mask_2= hair_mask.copy()
    hair_2 = cv2.bitwise_and(img,img,mask = hair_mask_2)

    for i in range(11):
        hair_mask = cv2.dilate(hair_mask, kernel)
    for i in range(10):
        hair_mask = cv2.erode(hair_mask, kernel)
    hair = cv2.bitwise_and(img,img,mask = hair_mask)

    def d(hair,hair_mask):
        #消除頭髮中255,255,255的部分
        mm = (hair == [255, 255, 255]).all(axis=2)
        hair[mm] = [0, 0, 0]
        hair_mask[mm] = 0

        return hair,hair_mask
    hair,hair_mask = d(hair,hair_mask)
    hair_2,hair_mask_2 = d(hair_2,hair_mask_2)
    result_hair,result_hair_mask = add_hair_bangs(hair,hair_mask,hair_2,img,shape)

    contours,hierarchy=cv2.findContours(hair_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    #取得交界處的點
    junction = {}
    for p in contours[0]:
        if(p[0][0] >= shape[0][0] and p[0][0] <= shape[16][0]):

            if p[0][0] in junction:
                if(junction[p[0][0]] < p[0][1]):
                    junction[p[0][0]] = p[0][1]
            else:
                junction[p[0][0]] = p[0][1]
    sorted_junction = dict(sorted(junction.items(), key=lambda x: x[0]))    #使用鍵值排序
    junction_array = [(key, value) for key, value in sorted_junction.items()]       #轉換成陣列

    #左邊眼睛到右邊眼睛點找最小的當作中間參考點
    y_values = [item[1] for item in junction_array[shape[17][0]:shape[26][0]+1]]

    min_y = min(y_values)
    min_index = shape[17][0] + y_values.index(min_y)
    mid_reference_point = (min_index,min_y)

    #頭髮參考點，第一個是頭髮左邊、第二個是頭髮右邊、第三個為頭髮中間
    reference_point = [junction_array[0],junction_array[-1],mid_reference_point]

    #定頭髮位的向量(到臉頰左右點)
    reference_vector = [(reference_point[0][0]-shape[0][0],reference_point[0][1]-shape[0][1]),
                        (reference_point[1][0]-shape[16][0],reference_point[1][1]-shape[16][1]),
                        (reference_point[2][0]-shape[0][0],reference_point[2][1]-shape[0][1])]


    #切出頭髮
    contours , _ = cv2.findContours(result_hair_mask,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area>max_area:
            x,y,w,h = cv2.boundingRect(c)
            max_area = area

    sub_hair = result_hair[y:y+h,x:x+w]
    sub_hair_mask = result_hair_mask[y:y+h,x:x+w]
    sub_reference_point = [(reference_point[0][0]-x,reference_point[0][1]-y),
                           (reference_point[1][0]-x,reference_point[1][1]-y),
                           (reference_point[2][0]-x,reference_point[2][1]-y)]


    return sub_hair,sub_hair_mask,reference_vector,sub_reference_point

def cut(img,shape):
    #生成mask
    left_eye_center = (shape[36][0]+(shape[39][0]-shape[36][0])//2,shape[38][1]+(shape[40][1]-shape[38][1])//2)
    right_eye_center = (shape[42][0]+(shape[45][0]-shape[42][0])//2,shape[43][1]+(shape[47][1]-shape[43][1])//2)
    mouse_center = (shape[48][0]+(shape[54][0]-shape[48][0])//2,shape[52][1]+(shape[57][1]-shape[52][1])//2)

    left_eye = img[left_eye_center[1]-15:left_eye_center[1]+25,left_eye_center[0]-20:left_eye_center[0]+20]
    right_eye = img[right_eye_center[1]-15:right_eye_center[1]+25,right_eye_center[0]-20:right_eye_center[0]+20]
    mouse = img[mouse_center[1]-20:mouse_center[1]+20,mouse_center[0]-30:mouse_center[0]+30]

    return left_eye,right_eye,mouse

def mix_img(shape,sketch,left_eye_sketch,right_eye_sketch,mouse_sketch):
    #讓眼睛對準中間
    left_eye_sketch = left_eye_sketch[0:30,:,:]
    left_eye_sketch_mask = 255*np.ones((left_eye_sketch.shape[0],left_eye_sketch.shape[1],1), left_eye_sketch.dtype)
    #讓眼睛對準中間
    right_eye_sketch = right_eye_sketch[0:30,:,:]
    right_eye_sketch_mask = 255*np.ones(right_eye_sketch.shape, right_eye_sketch.dtype)
    #畫像嘴巴
    mouse_mask = 255*np.ones(mouse_sketch.shape, mouse_sketch.dtype)

    left_eye_center = (shape[36][0]+(shape[39][0]-shape[36][0])//2,shape[38][1]+(shape[40][1]-shape[38][1])//2)
    right_eye_center = (shape[42][0]+(shape[45][0]-shape[42][0])//2,shape[43][1]+(shape[47][1]-shape[43][1])//2)
    mouse_center = (shape[48][0]+(shape[54][0]-shape[48][0])//2,shape[52][1]+(shape[57][1]-shape[52][1])//2)

    normal_clone = cv2.seamlessClone(src=left_eye_sketch, dst=sketch, mask=left_eye_sketch_mask, p=left_eye_center, flags=cv2.NORMAL_CLONE)
    normal_clone = cv2.seamlessClone(right_eye_sketch, normal_clone, right_eye_sketch_mask, right_eye_center, cv2.NORMAL_CLONE)
    normal_clone = cv2.seamlessClone(mouse_sketch, normal_clone, mouse_mask, mouse_center, cv2.NORMAL_CLONE)

    return normal_clone

def rotate_img(hair_img , hair_mask , rotate_value):
    #避免旋轉時角落超過陣列消失
    img_bg = np.zeros((256,256,3),dtype=np.uint8)
    mask_bg = np.zeros((256,256),dtype=np.uint8)
    p1 = ((256//2)-(hair_img.shape[1]//2),((256//2)-(hair_img.shape[0]//2)))
    p2 = [(256//2)+(hair_img.shape[1]//2),((256//2)+(hair_img.shape[0]//2))]
    if(hair_img.shape[1]%2 == 1) : p2=[p2[0]+1,p2[1]+0]
    if(hair_img.shape[0]%2 == 1) : p2=[p2[0]+0,p2[1]+1]
    img_bg[p1[1]:p2[1],p1[0]:p2[0]] = hair_img
    mask_bg[p1[1]:p2[1],p1[0]:p2[0]] = hair_mask

    #找旋轉中心(圖片中心)
    (h, w, d) = img_bg.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心

    #開始旋轉
    M = cv2.getRotationMatrix2D(center, int(rotate_value)*-1, 1.0)
    rotate_img = cv2.warpAffine(img_bg, M, (w, h))
    rotate_mask = cv2.warpAffine(mask_bg, M, (w, h))

    #縮小圖片
    contours,hierarchy = cv2.findContours(rotate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    rotate_img = rotate_img[y:y+h,x:x+w]
    rotate_mask = rotate_mask[y:y+h,x:x+w]

    return rotate_img,rotate_mask

#回傳啟起始座標、頭髮參考點、縮放過後的大小
def auto_adjust(face_shape,hair,reference_vector,sub_reference_point):
    #用reference_vector找到臉部的定位點
    face_reference_point = ((reference_vector[0][0]+face_shape[0][0],reference_vector[0][1]+face_shape[0][1]),
                            (reference_vector[1][0]+face_shape[16][0],reference_vector[1][1]+face_shape[16][1]),
                            (reference_vector[2][0]+face_shape[0][0],reference_vector[2][1]+face_shape[0][1]))

    #resize讓右邊點對準
    hair_distance = abs(sub_reference_point[0][0] - sub_reference_point[1][0])
    face_distance = abs(face_reference_point[0][0] - face_reference_point[1][0])

    scale_ratio = face_distance / hair_distance

    #縮放過後的圖片hair2
    hair_size = (int(hair.shape[1] * scale_ratio), int(hair.shape[0] * scale_ratio))#(長、寬)

    #更新頭髮參考點(因為resize過)
    sub_reference_point = ((int(sub_reference_point[0][0] * scale_ratio),int(sub_reference_point[0][1] * scale_ratio)),
                            (int(sub_reference_point[1][0] * scale_ratio),int(sub_reference_point[1][1] * scale_ratio)),
                            (int(sub_reference_point[2][0] * scale_ratio),int(sub_reference_point[2][1] * scale_ratio)))
    #print(sub_reference_point)
    #起始點(先以左邊點定位)
    start_point = (face_reference_point[0][0]-sub_reference_point[0][0],face_reference_point[2][1]-sub_reference_point[2][1])

    return start_point,sub_reference_point,hair_size