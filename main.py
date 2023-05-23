#usr/bin/python

from io import BytesIO
import json
import numpy as np
import uvicorn
from PIL import Image, ImageOps
import io
import tensorflow as tf
from keras.models import load_model
import PIL
import urllib.request
from fastapi import FastAPI, Request
from fastapi import UploadFile,File, HTTPException, Depends

from typing import Optional
from pydantic import BaseModel
import requests
import base64
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


model = load_model('new_keras_model.h5',compile=False)


def predict(image: Image.Image):
    #newlabels = ['Bathroom','Room-bedroom','Living_Room','Outdoor_building','Kitchen','Non_Related','Garden','Plot','Empty_room']
    labels = ['Bathroom','Bedroom','Living Room','Exterior View','Kitchen','Garden','Plot','Room','Swimming Pool','Gym','Parking','Map Location','Balcony','Floor Plan','Furnishing','Building Lobby','Office area','Stair','Master plan']

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

    

    '''
    prediction_dict = {
        "response": {
            "solutions": {
                "re_roomtype_eu_v2": {
                    "predictions": [
                        {
                            "confidence": str(result[0][0]),
                            "label": str(labels[0])
                        },
                        {
                            "confidence": str(result[0][1]),
                            "label": str(labels[1])
                        },
                        {
                            "confidence": str(result[0][2]),
                            "label": str(labels[2])
                        },
                        {
                            "confidence": str(result[0][3]),
                            "label": str(labels[3])
                        },
                        {
                            "confidence": str(result[0][4]),
                            "label": str(labels[4])
                        },
                        {
                            "confidence": str(result[0][5]),
                            "label": str(labels[5])
                        },
                        {
                            "confidence": str(result[0][6]),
                            "label": str(labels[6])
                        },
                        {
                            "confidence": str(result[0][7]),
                            "label": str(labels[7])
                        },
                        {
                            "confidence": str(result[0][8]),
                            "label": str(labels[8])
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
    '''
    return prediction_dict


@app.get("/predict_from_url")
async def predict_image(image_url: str):
    function = predict_image1(image_url)
    return function


def predict_image1(str_url: str):
    response = requests.get(str_url)
    print(response)
    image_bytes = io.BytesIO(response.content)
    print(image_bytes)
    img = PIL.Image.open(image_bytes)
    prediction1 = predict(img)
    data = json.dumps(prediction1)
    data1 = json.loads(data.replace("\'", '"'))
    return data1


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
