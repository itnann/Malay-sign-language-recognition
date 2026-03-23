import shutil
import os
import uuid
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
from typing import List
from contextlib import asynccontextmanager

from model import CustomLSTM
from utils import GESTURES, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, process_video_file

# --- 生命周期管理 ---
ml_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🔄 Loading BIM Model on {device}...")
    try:
        model = CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        ml_models["model"] = model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# 如果你有静态资源（比如CSS/Logo），可以挂载
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 数据模型 ---
class GestureInput(BaseModel):
    features: List[List[float]]


# --- 路由 ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. 使用指南页面
@app.get("/guide", response_class=HTMLResponse)
async def guide(request: Request):
    return templates.TemplateResponse("how_to_use.html", {"request": request})

# 3. 关于项目页面
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# 1. 实时流处理接口
@app.post("/predict_stream")
async def predict_stream(data: GestureInput):
    if "model" not in ml_models:
        return {"error": "Model not ready"}

    model = ml_models["model"]
    input_tensor = torch.tensor(data.features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return {
        "gesture": GESTURES[predicted_idx.item()],
        "confidence": f"{confidence.item():.2%}"
    }


# 2. 视频文件上传接口
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if "model" not in ml_models:
        return {"error": "Model not ready"}

    # 保存临时文件
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 调用 model_utils 中的处理函数
        input_tensor = process_video_file(temp_filename)

        if input_tensor is None:
            return {"error": "Could not extract features from video"}

        input_tensor = input_tensor.to(device)

        # 推理
        model = ml_models["model"]
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        result = {
            "gesture": GESTURES[predicted_idx.item()],
            "confidence": f"{confidence.item():.2%}"
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)