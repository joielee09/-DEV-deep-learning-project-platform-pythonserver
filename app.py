import io
from flask import Flask, jsonify, request
from PIL import Image
from flask_cors import CORS, cross_origin
import os
import json
import base64
import sys


# import modules
# computer vision
from imageClassification import classifier
# recommendation system
from contentbasedMovieRec import contentbased


app = Flask(__name__)
CORS(app)


@app.route("/")
def imageClassification():
  return 'landing page'


@app.route("/imageCls", methods=['POST'])
def imagefunc():
  content = request.get_json(force=True, silent=True)
  print("content:", content)
  if request.method == 'POST':    
    img = content['imageFile'] # get blob
    print("img", img)
    imgdata = base64.b64decode(img)
    filename = 'local_image.jpg'
    with open(filename, 'wb') as f:
      f.write(imgdata)
    image = Image.open('local_image.jpg')
    image = transforms_test(image).unsqueeze(0).to(device)

    class_name= classifier.imagepredict(image)
    print("result:", {'class_name': class_name})
    os.remove('./local_image.jpg')
    return jsonify({'class_name': class_name})


@app.route("/contentbasedMovieRec", methods=['POST'])
def contentbasedMovieRec():
  content = request.get_json(force=True, silent=True)
  print("title:", content['title'])
  title=content['title']
  year=content['year']
  input = title+' ('+year+')'
  print("input", input)
  if request.method == 'POST':
    res = contentbased.moviepredict(input)
    print("result:", res)
    return jsonify(res)


if (__name__) == "__main__":
  app.run(host='0.0.0.0', port=80)
