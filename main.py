# -*- coding: utf-8 -*-
import base64
import json
import os
from io import BytesIO

import numpy as np
import skimage
from flask import Flask, request, jsonify
from keras.models import load_model
from skimage import transform
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)
print('load model...')
model = load_model(dir_path + '/modle/captcha_model.h5')
print('load model...  success')
img_size = (130, 53)  # 全体图片都resize成这个尺寸
width, height, n_len = 130, 53, 4
batch_size = 32
t = {}
print('load done.')


def data_generator(imgageBase64):  # 样本生成器，节省内存
    image = base64_to_image(imgageBase64)
    x = np.zeros((1, width, height, 3), dtype=np.uint8)
    x[0] = np.array(transform.resize(image, img_size))
    return x


# 若img.save()报错 cannot write mode RGBA as JPEG
# 则img = Image.open(image_path).convert('RGB')
def image_to_base64(image_path):
    img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def base64_to_image(base64_string, image_path=None):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")
    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img


def process_captche(imgageBase64):
    global model
    data = data_generator(imgageBase64)
    pred_y = model.predict(data, verbose=1)
    pred_y = np.array([i.argmax(axis=1) for i in pred_y]).T
    list = [chr(i + 65) for i in pred_y[0]]
    return ''.join(list)


def test():
    # load 进来模型紧接着就执行一次 predict 函数
    base64Code = image_to_base64(dir_path + '/test/test.jpg')
    data = data_generator(base64Code)
    pred_y = model.predict(data)
    pred_y = np.array([i.argmax(axis=1) for i in pred_y]).T
    list = [chr(i + 65) for i in pred_y[0]]
    print(list)
    print('test done.')


@app.route("/captcha", methods=['POST'])
def captcha():
    if request.method == "POST":
        # name = request.args.get('uuid')
        name = "test"
        dataStr = request.get_data()
        dataStr = dataStr.decode("utf-8")
        jsonData = json.loads(dataStr)
        str = jsonData['captchaData'].replace(' ', '+')
        try:
            # imgdata = base64.b64decode(str)
            data = {}
            data["recognition"] = process_captche(str)
            t['data'] = data
            t['code'] = 200
            t['message'] = "success"
            print(t)
            return jsonify(t)
        except Exception as e:
            t['data'] = ""
            t['code'] = 200
            t['message'] = "error"
            return jsonify(t)
    else:
        t['data'] = ""
        t['code'] = "400"
        t['message'] = "GET NOT BE SUPPORT"
        return jsonify(t)


if __name__ == '__main__':
    test()
    app.run(host='127.0.0.1', port=5000, debug=True)
