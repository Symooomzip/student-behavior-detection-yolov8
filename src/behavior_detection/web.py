"""Local FastAPI application backing the React workspace."""
import io
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .config import load_config, resolve_path
from .inference import Detector, LABELS, MIN_CONFIDENCE
from .video_jobs import VideoJobs

IMAGE_LIMIT = 25 * 1024 * 1024
VIDEO_LIMIT = 200 * 1024 * 1024


def decode_image(content):
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("L’image dépasse la limite de 25 mégapixels.")
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image illisible. Utilisez une image JPG, PNG ou WebP valide.") from exc


def create_app(cfg=None, weights=None, device=None, detector=None, frontend_dir=None):
    cfg = cfg or load_config()
    root = Path(cfg["_root"])
    weights_path = Path(weights) if weights else resolve_path(cfg, "weights")
    if weights is None and not weights_path.exists():
        legacy = root / "models" / "best_legacy_run.pt"
        if legacy.exists():
            weights_path = legacy
    engine = detector or Detector(weights_path, cfg["classes"], device=device)

    @asynccontextmanager
    async def lifespan(app):
        with tempfile.TemporaryDirectory(prefix="student-behavior-") as temp:
            app.state.temp = Path(temp)
            app.state.jobs = VideoJobs(temp, engine)
            try:
                yield
            finally:
                await run_in_threadpool(app.state.jobs.close)

    app = FastAPI(title="Student Behavior Detection", version="0.3.0", lifespan=lifespan)

    @app.get("/api/meta")
    def meta():
        return {"classes": [{"id": i, "name": name, "label": LABELS.get(name, name)}
                            for i, name in enumerate(cfg["classes"])],
            "model": weights_path.name, "model_available": weights_path.is_file(),
            "legacy": weights_path.name == "best_legacy_run.pt",
            "confidence": cfg["app"]["conf"], "iou": cfg["app"]["iou"],
            "min_confidence": MIN_CONFIDENCE, "image_limit_mb": 25, "video_limit_mb": 200,
            "video_limit_seconds": 600,
            "example_available": (root / "data/raw/images/image001.jpg").exists()}

    @app.get("/api/example")
    def example():
        path = root / "data/raw/images/image001.jpg"
        if not path.is_file():
            raise HTTPException(404, "L’exemple n’est pas disponible dans cette installation.")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/image")
    async def image(file: UploadFile = File(...), iou: float = Query(0.45, ge=0.1, le=0.9)):
        try:
            content = await file.read(IMAGE_LIMIT + 1)
            if len(content) > IMAGE_LIMIT:
                raise HTTPException(413, "L’image dépasse 25 Mo.")
            frame = await run_in_threadpool(decode_image, content)
            return await run_in_threadpool(engine.predict, frame, iou=iou)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Image inference failed")
            raise HTTPException(500, "L’analyse a échoué. Consultez le terminal du serveur.") from exc
        finally:
            await file.close()

    @app.post("/api/videos", status_code=202)
    async def video(file: UploadFile = File(...), sample_fps: float = Query(4, ge=1, le=10),
                    iou: float = Query(0.45, ge=0.1, le=0.9),
                    confidence: float = Query(0.25, ge=0.1, le=0.95)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}:
            await file.close()
            raise HTTPException(415, "Format non pris en charge. Utilisez MP4, MOV, AVI, WebM ou MKV.")
        path = None
        try:
            with tempfile.NamedTemporaryFile(dir=app.state.temp, suffix=suffix, delete=False) as dest:
                path = Path(dest.name)
                size = 0
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > VIDEO_LIMIT:
                        raise HTTPException(413, "La vidéo dépasse 200 Mo.")
                    dest.write(chunk)
            return app.state.jobs.create(path, Path(file.filename).name, sample_fps, iou, confidence)
        except ValueError as exc:
            raise HTTPException(429, str(exc)) from exc
        finally:
            await file.close()
            if path:
                path.unlink(missing_ok=True)

    def get_job(key):
        try:
            return app.state.jobs.status(key)
        except KeyError as exc:
            raise HTTPException(404, "Analyse introuvable ou expirée. Relancez le traitement.") from exc

    @app.get("/api/videos/{key}")
    def video_status(key: str):
        return get_job(key)

    @app.delete("/api/videos/{key}")
    def cancel_video(key: str):
        get_job(key)
        return app.state.jobs.cancel(key)

    @app.get("/api/videos/{key}/results")
    def video_results(key: str):
        if get_job(key)["state"] != "complete":
            raise HTTPException(409, "L’analyse n’est pas terminée.")
        return json.loads((app.state.temp / key / "results.json").read_text(encoding="utf-8"))

    @app.get("/api/videos/{key}/{kind}.mp4")
    def video_file(key: str, kind: str):
        if kind not in {"preview", "annotated"}:
            raise HTTPException(404)
        if get_job(key)["state"] != "complete":
            raise HTTPException(409, "L’analyse n’est pas terminée.")
        return FileResponse(app.state.temp / key / f"{kind}.mp4", media_type="video/mp4",
                            filename="student-behavior-annotated.mp4" if kind == "annotated" else None)

    @app.get("/api/{path:path}", include_in_schema=False)
    def missing_api(path: str):
        raise HTTPException(404, "Route API inconnue.")

    dist = Path(frontend_dir) if frontend_dir else root / "frontend" / "dist"
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def build_required():
            return JSONResponse({"message": "Dans frontend/, exécutez npm install puis npm run build, ou npm run dev."})
    return app
