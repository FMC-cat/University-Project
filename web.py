import os,pathlib,base64,cv2
import numpy as np
from flask import Flask, url_for, redirect,  render_template, request , make_response
from function.auto import *
# 取得目前檔案所在的資料夾
SRC_PATH =  pathlib.Path(__file__).parent.absolute()
INPUT_FOLDER = os.path.join(SRC_PATH,  'static', 'ipnut')
OUTPUT_FOLDER = os.path.join(SRC_PATH,  'static', 'output')

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():

    print("index")
    return render_template('create.html')

@app.route('/', methods=['POST'])
def upload_file():
    print("upload_file")
    file = request.files['filename']
    if file.filename != '':
        file_bytes = np.fromfile(file, np.uint8)
        global face_img,face_shape
        face_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        face_img,face_shape = get_256x256_img(face_img)
        retval, buffer = cv2.imencode('.jpg', face_img)
        base64_img = base64.b64encode(buffer).decode()

    return render_template("create.html",img_stream = base64_img)

@app.route('/return_hair', methods=['POST'])
def return_hair():
    if request.method == 'POST':
        select_img_url = os.path.join(SRC_PATH,  request.form['select_img'])
        #提取頭髮並加密成base64回傳

        select_img = cv2.imread(select_img_url)
        

        #取得hair圖的68點和變成256*256
        hair_img,hair_shape = get_256x256_img(select_img)
        #hair,hair_mask,hair_alignment_point = get_hair(select_img)舊的

        global bald,hair,hair_mask
        bald,hair,hair_mask,start_point,hair_size = auto_get_hair_pos(face_img,face_shape,hair_img,hair_shape)

        retval, buffer = cv2.imencode('.png', hair)
        base64_hair_img = base64.b64encode(buffer).decode("utf-8")
        base64_hair_img = "data:image/png;base64," + base64_hair_img

        #將剛剛獲得的沒頭髮圖片加密
        retval, buffer = cv2.imencode('.jpg', bald)
        base64_no_hair_img = base64.b64encode(buffer).decode()
        base64_no_hair_img = "data:image/jpg;base64," + base64_no_hair_img

        jason = {"hair_img":base64_hair_img,
                 "max_x" : 256-hair.shape[1],
                 "max_y" : 256-hair.shape[0],
                 "hair_img_x_size":hair.shape[1],
                 "hair_img_y_size":hair.shape[0],
                 "no_hair_img" : base64_no_hair_img,
                 "x" : str(start_point[0]),
                 "y" : str(start_point[1])
                }

        return jason
    else :
        return render_template("create.html")

#完成設定
@app.route('/receive_all_data', methods=['POST'])
def receive_all_data():
    if request.method == 'POST':
        img = merge_img(bald,hair,hair_mask,request.form['pos_x'],request.form['pos_y'],request.form['img_width'],request.form['img_height'] , request.form['rotate_value'])
        sketch = change_style(img , face_shape)

        #img 編碼成 base64
        retval, buffer = cv2.imencode('.jpg', img)
        base64_img = base64.b64encode(buffer).decode("utf-8")

        #sketch 編碼成 base64
        retval, buffer = cv2.imencode('.jpg', sketch)
        base64_sketch_img = base64.b64encode(buffer).decode("utf-8")

        jason = {"ori_img":base64_img,
                 "sketch_img":base64_sketch_img}

        print("全部完成")
        return jason

def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.frombuffer(imgString,np.uint8)
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return image

if __name__ == "__main__":
    app.run()
    # img = cv2.imread('D:/web2/static/image/hair/2.jpg')
    # get_hair(img)