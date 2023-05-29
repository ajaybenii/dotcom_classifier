#usr/bin/python
import io
import uvicorn
import json
import requests

import tensorflow as tf
import numpy as np

from io import BytesIO
from PIL import Image, ImageOps
from keras.models import load_model
from fastapi import FastAPI, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


origins = [
    "https://ai.propvr.tech",
    "http://ai.propvr.tech",
    "https://ai.propvr.tech/classify",
    "http://ai.propvr.tech/classify" 
    ]


'''
origins = [
    "https://ai.propvr.tech/classify",
    "http://ai.propvr.tech/classify",
    "https://getedge.glitch.me/*",
    "http://getedge.glitch.me/*",
    "https://getedge.glitch.me",
    "http://getedge.glitch.me" 
    ]'''


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return "Server is up!"


@app.get("/health")
async def root1():
    return "Server is up!"


model = load_model('update_keras_model.h5',compile=False)


def predict1(image: Image.Image):

    labels = ['Bathroom','Bedroom','Living Room','Exterior View','Kitchen','Garden','Plot','Room','Swimming Pool','Gym','Parking','Map Location','Balcony','Floor Plan','Furnished Amenities','Building Lobby','Team Area','Staircase','Master Plan']
    data = np.ndarray(shape=(1,224, 224, 3), dtype=np.float32)
    size = (224, 224)

    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    #image.show()
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    img_array_final = normalized_image_array[:, :, :3]
    #print("shape: ", img_array_final.shape)
    data[0] = img_array_final
    result = model.predict(data)
    #result1 = sorted(result, reverse=True)
    
    arr_sorted = -np.sort(-result,axis=1)
    top_five = arr_sorted[:,:5]
    top_five_array = result.argsort()
    top_five_array1 = top_five_array.tolist()
    top1 = top_five_array1[0][-1]
    top2 = top_five_array1[0][-2]
    top3 = top_five_array1[0][-3]
    top4 = top_five_array1[0][-4]
    top5 = top_five_array1[0][-5]
    
    #print(top_five)
    #max1 = np.max(result)
    index_max = np.argmax(result)
    #print(index_max)

    
    prediction_dict = {
        "response": {
            "solutions": {
                "re_roomtype_eu_v2": {
                    "predictions": [
                        {
                            "confidence": str(top_five[0][0]),
                            "label": str(labels[top1])
                        },
                        {
                            "confidence": str(top_five[0][1]),
                            "label": str(labels[top2])
                        },
                        {
                            "confidence": str(top_five[0][2]),
                            "label": str(labels[top3])
                        },
                        {
                            "confidence": str(top_five[0][3]),
                            "label": str(labels[top4])
                        },
                        {
                            "confidence": str(top_five[0][4]),
                            "label": str(labels[top5])
                        }
                    ],
                    "top_prediction":{
                        "confidence": str(result[0][index_max]),
                            "label": str(labels[index_max])
                    }
                }
            }
        }
    }

    return prediction_dict


@app.get("/predict_from_url")
async def predict_image(image_url: str):
    function = predict_image1(image_url)
    return function


@app.post("/upload")
async def upload_file(file: UploadFile=File(...)):
    '''This function is upload image from your system'''
    try:
       contents = await file.read()
       image_bytes = Image.open(BytesIO(contents))
    
    except:
      return("image not readable")
    
    
    function = predict1(image_bytes)
    return function


def predict_image1(str_url: str):
    response = requests.get(str_url)

    image_bytes = io.BytesIO(response.content)

    img = Image.open(image_bytes)
    prediction1 = predict1(img)
    data = json.dumps(prediction1)
    data1 = json.loads(data.replace("\'", '"'))
    return data1


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
