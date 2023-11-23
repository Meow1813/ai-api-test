import requests
import uvicorn
import validators
from PIL import Image
from fastapi import FastAPI, HTTPException
from starlette import status
from transformers import ViTImageProcessor, ViTForImageClassification

app = FastAPI()


@app.get("/get-photo-class")
async def get_image(url):
    if validators.url(url) is True:
        image = Image.open(requests.get(url, stream=True).raw)

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        return {"Predicted class:", model.config.id2label[predicted_class_idx]}
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid URL")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
