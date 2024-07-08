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

app = FastAPI()

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
