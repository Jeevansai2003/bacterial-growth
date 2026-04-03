from fastapi import APIRouter, UploadFile, File, Form
import os
import uuid
import logging

from app.services.segmentation import analyze_image

router = APIRouter()

logging.basicConfig(level=logging.INFO)


@router.post("/analyze/image")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    resize_factor: float = Form(1.0),
    manual_threshold: float = Form(0.5),
    use_watershed_split: bool = Form(True),
    method: str = Form("CW-MTF (Novel)"),
):
    try:
        # ---------------- SAVE FILE ----------------
        os.makedirs("uploads", exist_ok=True)

        file_ext = os.path.splitext(file.filename)[1]
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join("uploads", file_name)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # ---------------- DEBUG LOG ----------------
        logging.info(f"Selected Method: {method}")

        # ---------------- METHOD VALIDATION ----------------
        valid_methods = [
            "Otsu",
            "Adaptive",
            "Manual",
            "Majority Fusion",
            "CW-MTF (Novel)"
        ]

        if method not in valid_methods:
            return {"error": f"Invalid method: {method}"}

        # ---------------- METHOD EXECUTION ----------------
        # (Handled inside analyze_image, but logging here for debugging)

        if method == "Otsu":
            logging.info("Executing Otsu Segmentation")

        elif method == "Adaptive":
            logging.info("Executing Adaptive Segmentation")

        elif method == "Manual":
            logging.info(f"Executing Manual Segmentation (threshold={manual_threshold})")

        elif method == "Majority Fusion":
            logging.info("Executing Majority Fusion")

        elif method == "CW-MTF (Novel)":
            logging.info("Executing CW-MTF (Novel)")

        # ---------------- CALL CORE PIPELINE ----------------
        result = analyze_image(
            image_path=file_path,
            method=method,  # ✅ IMPORTANT FIX
            resize_factor=resize_factor,
            manual_threshold=manual_threshold,
            use_watershed_split=use_watershed_split,
        )

        # ---------------- FORMAT RESPONSE ----------------
        images = {
            "resized": result["images"]["resized"],
            "grayscale": result["images"]["gray"],
            "preprocessed": result["images"]["preprocessed"],
            "color_likelihood": result["images"]["color"],
            "binary": result["images"]["binary"],
            "refined_split": result["images"]["refined"],
            "segmented": result["images"]["segmented"],
        }

        # ✅ Add uncertainty ONLY for CW-MTF
        if method == "CW-MTF (Novel)" and "uncertainty" in result["images"]:
            images["uncertainty"] = result["images"]["uncertainty"]

        return {
            "id": result["id"],
            "method": method,
            "metrics": result["metrics"],
            "weights": result.get("weights", None),
            "images": images
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}