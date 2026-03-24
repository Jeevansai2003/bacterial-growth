from fastapi import APIRouter, UploadFile, File, Form
import os
import uuid

from app.services.segmentation import analyze_image

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/analyze/image")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    resize_factor: float = Form(1.0),
    manual_threshold: float = Form(0.5),
    use_watershed_split: bool = Form(True),
):
    os.makedirs("uploads", exist_ok=True)

    file_ext = os.path.splitext(file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join("uploads", file_name)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = analyze_image(
        image_path=file_path,
        resize_factor=resize_factor,
        manual_threshold=manual_threshold,
        use_watershed_split=use_watershed_split
    )

    return result