from fastapi import FastAPI

app = FastAPI(title="PaddleOCR API", description="OCR service using PaddleOCR", version="1.0.0")

# 挂载控制器路由
try:
    from app.controllers.ocr_controller import router as ocr_router
    app.include_router(ocr_router)
except Exception:
    # 避免导入阶段的循环依赖导致失败
    pass


