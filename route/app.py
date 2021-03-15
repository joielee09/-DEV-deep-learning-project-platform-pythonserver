import io
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request
from PIL import Image
from flask_cors import CORS, cross_origin
import os
import json
import base64


# import modules
import sys
# computer vision
sys.path.append('D:\__pythonserver\projects\computerVision\imageClassification')
import imageClassifier
# recommendation system
sys.path.append('D:\__pythonserver\projects\RecSys\ContentbasedMovieRec')
import contentBased


app = Flask(__name__)
CORS(app)


@app.route("/")
def imageClassification():
  return 'landing page'

@app.route("/computervision/project1", methods=['POST'])
def cv_project1():
  content = request.get_json(force=True, silent=True)
  print("content:", content)
  if request.method == 'POST':    
    # 이미지 바이트 데이터 받아오기
    img = content['imageFile'] # get blob
    print("img", img)
    imgdata = base64.b64decode(img)
    filename = 'local_image.jpg'
    with open(filename, 'wb') as f:
      f.write(imgdata)
    image = Image.open('local_image.jpg')
    image = transforms_test(image).unsqueeze(0).to(device)

    class_name= imageClassifier.get_prediction(image)
    print("결과:", {'class_name': class_name})
    os.remove('./local_image.jpg')
    return jsonify({'class_name': class_name})

@app.route("/recsys/project1", methods=['POST'])
def recsys_project1():
  content = request.get_json(force=True, silent=True)
  print("title:", content['title'])
  title=content['title']
  year=content['year']
  input = title+' ('+year+')'
  print("input", input)
  #processing title
  if request.method == 'POST':
    #받은 데이터 처리
    res = contentbased_valid.get_prediction(input)
    print("결과:", res)
    return jsonify(res)

if (__name__) == "__main__":
  run_with_ngrok(app)
  app.run()