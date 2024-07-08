# import torch
# import gradio as gr
# from read import text_recognizer
# from model import Model
# from utils import CTCLabelConverter
# from ultralytics import YOLO
# from PIL import ImageDraw

# """ vocab / character number configuration """
# file = open("UrduGlyphs.txt","r",encoding="utf-8")
# content = file.readlines()
# content = ''.join([str(elem).strip('\n') for elem in content])
# content = content+" "
# """ model configuration """
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# converter = CTCLabelConverter(content)
# recognition_model = Model(num_class=len(converter.character), device=device)
# recognition_model = recognition_model.to(device)
# recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# recognition_model.eval()

# detection_model = YOLO("yolov8m_UrduDoc.pt")

# examples = ["1.jpg","2.jpg","3.jpg"]

# input = gr.Image(type="pil",image_mode="RGB", label="Input Image")

# def predict(input):
#     "Line Detection"
#     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
#     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
#     bounding_boxes.sort(key=lambda x: x[1])
    
#     "Draw the bounding boxes"
#     draw = ImageDraw.Draw(input)
#     for box in bounding_boxes:
#         # draw rectangle outline with random color and width=5
#         from numpy import random
#         draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
#     "Crop the detected lines"
#     cropped_images = []
#     for box in bounding_boxes:
#         cropped_images.append(input.crop(box))
#     len(cropped_images)
    
#     "Recognize the text"
#     texts = []
#     for img in cropped_images:
#         texts.append(text_recognizer(img, recognition_model, converter, device))
    
#     "Join the text"
#     text = "\n".join(texts)
    
#     "Return the image with bounding boxes and the text"
#     return input,text

# output_image = gr.Image(type="pil",image_mode="RGB",label="Detected Lines")
# output_text = gr.Textbox(label="Recognized Text",interactive=True,show_copy_button=True)

# iface = gr.Interface(predict,
#                      inputs=input,
#                      outputs=[output_image,output_text],
#                      title="End-to-End Urdu OCR",
#                      description="Demo Web App For UTRNet\n(https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)",
#                      examples=examples,
#                      allow_flagging="never")
# iface.launch()































#==================================================





# import torch
# import gradio as gr
# from read import text_recognizer
# from model import Model
# from utils import CTCLabelConverter
# from ultralytics import YOLO
# from deep_translator import GoogleTranslator  # Add the GoogleTranslator module
# from PIL import Image

# # Load Urdu glyphs
# with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
#     content = file.read().replace('\n', '') + " "

# # Model configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# converter = CTCLabelConverter(content)
# recognition_model = Model(num_class=len(converter.character), device=device)
# recognition_model = recognition_model.to(device)
# recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# recognition_model.eval()

# detection_model = YOLO("yolov8m_UrduDoc.pt")

# examples = ["1.jpg", "2.jpg", "3.jpg"]
# input_image = gr.Image(type="pil", image_mode="RGB", label="Input Image")

# def predict(input):
#     # Line Detection
#     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
#     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
#     bounding_boxes.sort(key=lambda x: x[1])
    
#     # Crop the detected lines
#     cropped_images = [input.crop(box) for box in bounding_boxes]

#     # Recognize the text
#     texts = [text_recognizer(img, recognition_model, converter, device) for img in cropped_images]
    
#     # Join the text
#     urdu_text = "\n".join(texts)
    
#     # Translate to English
#     translated_text = GoogleTranslator(source='ur', target='en').translate(urdu_text)
    
#     # Return the original image (without bounding boxes) and the translated text
#     return input, translated_text

# output_image = gr.Image(type="pil", image_mode="RGB", label="Detected Lines")
# output_text = gr.Textbox(label="Recognized Text", interactive=True, show_copy_button=True)

# iface = gr.Interface(
#     predict,
#     inputs=input_image,
#     outputs=[output_image, output_text],
#     title="End-to-End Urdu OCR",
#     description="Demo Web App For UTRNet\n(https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)",
#     examples=examples,
#     allow_flagging="never"
# )

# iface.launch()





















# import torch
# import gradio as gr
# from read import text_recognizer
# from model import Model
# from utils import CTCLabelConverter
# from ultralytics import YOLO
# from deep_translator import GoogleTranslator  # Add the GoogleTranslator module
# from PIL import Image

# # Load Urdu glyphs
# with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
#     content = file.read().replace('\n', '') + " "

# # Model configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# converter = CTCLabelConverter(content)
# recognition_model = Model(num_class=len(converter.character), device=device)
# recognition_model = recognition_model.to(device)
# recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# recognition_model.eval()

# detection_model = YOLO("yolov8m_UrduDoc.pt")

# examples = ["1.jpg", "2.jpg", "3.jpg"]
# input_image = gr.Image(type="pil", image_mode="RGB", label="Input Image")

# def predict(input):
#     # Line Detection
#     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
#     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
#     bounding_boxes.sort(key=lambda x: x[1])
    
#     # Crop the detected lines
#     cropped_images = [input.crop(box) for box in bounding_boxes]

#     # Recognize the text
#     texts = [text_recognizer(img, recognition_model, converter, device) for img in cropped_images]
    
#     # Join the text
#     urdu_text = "\n".join(texts)
    
#     # Translate to English
#     translated_text = GoogleTranslator(source='ur', target='en').translate(urdu_text)
    
#     # Return the original image, the recognized Urdu text, and the translated English text
#     return input, urdu_text, translated_text

# output_image = gr.Image(type="pil", image_mode="RGB", label="Input Image")
# output_urdu_text = gr.Textbox(label="Recognized Urdu Text", interactive=True, show_copy_button=True)
# output_translated_text = gr.Textbox(label="Translated English Text", interactive=True, show_copy_button=True)

# iface = gr.Interface(
#     predict,
#     inputs=input_image,
#     outputs=[output_image, output_urdu_text, output_translated_text],
#     title="End-to-End Urdu OCR",
#     description="Urdu OCR",
#     examples=examples,
#     allow_flagging="never"
# )

# iface.launch()

























########################################### FAST API


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import io

app = FastAPI(
    title="End-to-End Urdu OCR for CNIC",
    description="This will take the image on CNIC and return the text on it in Urdu and than translate it to English."
)

# Load Urdu glyphs
with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
    content = file.read().replace('\n', '') + " "

# Model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
recognition_model.eval()

detection_model = YOLO("yolov8m_UrduDoc.pt")

@app.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    # Load image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Line Detection
    detection_results = detection_model.predict(source=image, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])

    # Crop the detected lines
    cropped_images = [image.crop(box) for box in bounding_boxes]

    # Recognize the text
    texts = [text_recognizer(img, recognition_model, converter, device) for img in cropped_images]

    # Join the text
    urdu_text = "\n".join(texts)

    # Translate to English
    translated_text = GoogleTranslator(source='ur', target='en').translate(urdu_text)

    return JSONResponse(content={"urdu_text": urdu_text, "translated_text": translated_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
