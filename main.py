

from PIL import Image
import logging
# import requests
import cv2
import os, sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir=""):
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  return model

# Define a function to solve an image similarity puzzle
def solve_puzzle(query_imagel, candidate_images, model):
  # Preprocess the query image and the candidate images using the processor
  objects=["bicycle", "chiar","window", "ball", "ship","tv screen","bag","cat", "phone", "cow", "hammer", "fox", "dog", "umbrella","Watering can"]

  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  query_image = processor(text=objects,images=query_imagel, return_tensors="pt", padding=True)
  query_caption = model(**query_image)
  probs = query_caption.logits_per_image.softmax(dim=1)
  index= probs[0].argmax()
  # print(probs[0])
  print(objects[index])
  candidate_images=[
      processor(text=objects,images=c_image, return_tensors="pt", padding=True) for c_image in candidate_images
  ]

  query_caption = model(**query_image)
  candidate_captions = [
      float(model(**candidate_image).logits_per_image.softmax(dim=1)[0][index]) for candidate_image in candidate_images
  ]
  # print(candidate_captions)
  return candidate_captions

# Import the necessary modules


# def solve_captchas():
#   images = [file for file in os.listdir(folder) if file.endswith('.jpg') or file.endswith('.JPG')]
#   if len(images) == 0:
#     print("No images found : Add the captchas to the \"images\" folder")
#     return
#   # Split each image into five equal parts horizontally and vertically
#   # print(images)
#   for image in images:
#     print(image)
#     img = cv2.cvtColor(cv2.imread(folder + image), cv2.COLOR_BGR2RGB)

#     # Read the image
#     # img = cv2.imread('/content/drafar05.jpg')

#     # Split the image into five equal parts horizontally and vertically
#     h, w = img.shape[:2]
#     img1 = img[int(h // (30)):int(h // (2.6)), int(w // (2.42)):int(w // (1.71))]
#     img2 = img[int(h // (1.8)):int(h // (1.05)), int(w // (8.5)):int(w // (3.3))]
#     img3 = img[int(h // (1.8)):int(h // (1.05)), int(w // (3.3)): int(w // (2.0))]
#     img4 = img[int(h // (1.8)):int(h // (1.05)), int(w // (2.0)):  int(w // (1.445))]
#     img5 = img[int(h // (1.8)):int(h // (1.05)),  int(w // (1.445)):int(w // (1.13))]

#     query_image=Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
#     candidate_images=[
#         Image.fromarray(img2),
#         Image.fromarray(img3),
#         Image.fromarray(img4),
#         Image.fromarray(img5)
#     ]
#     best_image = solve_puzzle(query_image, candidate_images)
#     print("captcha answer: image ",best_image.index(max(best_image))+1)

def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    image_data = json.loads(request_body)["image_data"]

    # Convert the image data to a numpy array
    image_array = np.fromstring(image_data, np.uint8)
    
    return image_array


# inference
def predict_fn(input_object, model):
    img = cv2.imdecode(input_object, cv2.COLOR_BGR2RGB)

    # Read the image
    # img = cv2.imread('/content/drafar05.jpg')

    # Split the image into five equal parts horizontally and vertically
    h, w = img.shape[:2]
    img1 = img[int(h // (30)):int(h // (2.6)), int(w // (2.42)):int(w // (1.71))]
    img2 = img[int(h // (1.8)):int(h // (1.05)), int(w // (8.5)):int(w // (3.3))]
    img3 = img[int(h // (1.8)):int(h // (1.05)), int(w // (3.3)): int(w // (2.0))]
    img4 = img[int(h // (1.8)):int(h // (1.05)), int(w // (2.0)):  int(w // (1.445))]
    img5 = img[int(h // (1.8)):int(h // (1.05)),  int(w // (1.445)):int(w // (1.13))]

    query_image=Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    candidate_images=[
        Image.fromarray(img2),
        Image.fromarray(img3),
        Image.fromarray(img4),
        Image.fromarray(img5)
    ]
    best_image = solve_puzzle(query_image, candidate_images, model)
    return best_image


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    return json.dumps(predictions)