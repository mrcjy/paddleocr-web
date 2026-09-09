"""PaddleOCR service: upload images, queue them, run OCR, return results."""
import json
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

DATA_DIR = Path("/app/data")
JOBS_FILE = DATA_DIR / "jobs.json"
UPLOAD_DIR = DATA_DIR / "uploads"
THUMB_DIR = DATA_DIR / "thumbs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = int(__import__("os").environ.get("OCR_WORKERS", "1"))

app = FastAPI(title="PaddleOCR Web")

_lock = threading.Lock()


def _load_jobs():
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_jobs(jobs):
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(JOBS_FILE)


_jobs: dict = _load_jobs()
_q: "queue.Queue[str]" = queue.Queue()

# ---------------------------------------------------------------- job store

def _update(job_id: str, **fields):
    with _lock:
        _jobs[job_id].update(fields)
        _save_jobs(_jobs)


def _queue_position(job_id: str) -> int:
    """1-based position in the pending queue, 0 if not queued."""
    with _q.mutex:
        for i, jid in enumerate(list(_q.queue)):
            if jid == job_id:
                return i + 1
    return 0


# ---------------------------------------------------------------- worker

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return _ocr


def _run_ocr(job_id: str, path: Path):
    try:
        result = _get_ocr().ocr(str(path), cls=True)
        lines = []
        for page in result or []:
            for item in page or []:
                box, (text, conf) = item
                lines.append({"text": text, "confidence": round(float(conf), 4),
                              "box": [[float(x), float(y)] for x, y in box]})
        text_all = "\n".join(l["text"] for l in lines)
        _update(job_id, status="done", finished_at=time.time(),
                lines=lines, text=text_all)
    except Exception as e:  # noqa: BLE001
        _update(job_id, status="error", error=str(e), finished_at=time.time())
    finally:
        path.unlink(missing_ok=True)


def _worker():
    while True:
        job_id = _q.get()
        with _lock:
            job = _jobs.get(job_id)
        if not job or job["status"] != "queued":
            continue
        path = UPLOAD_DIR / job["filename"]
        _update(job_id, status="processing", started_at=time.time())
        _run_ocr(job_id, path)


for _ in range(WORKERS):
    threading.Thread(target=_worker, daemon=True).start()

# ---------------------------------------------------------------- api

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@app.get("/api/health")
def health():
    return {"ok": True, "pending": _q.qsize(), "jobs": len(_jobs)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"不支持的文件类型: {suffix or '(无后缀)'}")
    job_id = uuid.uuid4().hex[:12]
    filename = f"{job_id}{suffix}"
    path = UPLOAD_DIR / filename
    path.write_bytes(await file.read())

    # 生成缩略图用于前端预览（原图处理完即删，缩略图保留）
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((720, 720))
            im.save(THUMB_DIR / f"{job_id}.jpg", quality=82)
    except Exception:
        pass

    with _lock:
        _jobs[job_id] = {
            "id": job_id, "name": file.filename or filename,
            "filename": filename, "status": "queued",
            "created_at": time.time(),
        }
        _save_jobs(_jobs)
    _q.put(job_id)
    return {"id": job_id, "status": "queued", "position": _q.qsize()}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        job = dict(job)
    if job["status"] == "queued":
        job["position"] = _queue_position(job_id)
    return JSONResponse(job)


@app.get("/api/image/{job_id}")
def get_image(job_id: str):
    thumb = THUMB_DIR / f"{job_id}.jpg"
    if not thumb.exists():
        raise HTTPException(404, "图片不存在")
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
        jobs = [dict(j) for j in jobs[:50]]
    for j in jobs:
        if j["status"] == "queued":
            j["position"] = _queue_position(j["id"])
        j.pop("lines", None)  # 逐行坐标数据量大，列表页不带；单任务接口里仍有
        j["has_image"] = (THUMB_DIR / f"{j['id']}.jpg").exists()
    return jobs


@app.get("/")
def index():
    return FileResponse("/app/static/index.html")
