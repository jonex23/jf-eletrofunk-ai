import json, os, shutil, threading, time, uuid
from pathlib import Path
import numpy as np, soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import shutil as _sh
import engine, generative

HERE = Path(__file__).resolve().parent
HAS_FFMPEG = _sh.which("ffmpeg") is not None

MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 50))
TTL = int(os.environ.get("JOB_TTL_MIN", 30)) * 60
ALLOWED = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma"}
WORK = Path(os.environ.get("WORK_DIR", str(Path(__import__("tempfile").gettempdir()) / "jf-eletrofunk"))); WORK.mkdir(parents=True, exist_ok=True)
JOBS = {}
app = FastAPI(title="JF ELETROFUNK AI")

def sweep():  # exclui arquivos temporários expirados
    while True:
        for jid, j in list(JOBS.items()):
            if time.time() - j["t"] > TTL: shutil.rmtree(WORK / jid, ignore_errors=True); JOBS.pop(jid, None)
        time.sleep(60)
threading.Thread(target=sweep, daemon=True).start()

async def save_upload(file: UploadFile, d: Path):
    if not HAS_FFMPEG: raise HTTPException(500, "ffmpeg não está instalado no servidor. Use o Dockerfile ou instale o ffmpeg.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400, f"Formato {ext or 'desconhecido'} não suportado. Use MP3, WAV, FLAC ou M4A.")
    raw = d / f"upload{ext}"; size = 0
    with open(raw, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20: raw.unlink(); raise HTTPException(413, f"Arquivo maior que {MAX_MB} MB.")
            f.write(chunk)
    try: engine.to_wav(str(raw), str(d / "original.wav"))
    except ValueError as e: raise HTTPException(400, str(e))
    finally: raw.unlink(missing_ok=True)

@app.get("/api/health")
def health():
    return {"ok": True, "ffmpeg": HAS_FFMPEG}

@app.get("/api/config")
def config():
    return {"max_mb": MAX_MB, "ffmpeg": HAS_FFMPEG, "generative": generative.available(),
            "presets": {k: {"label": v[0], "style": v[1], "bpm": v[2], "bass": v[3], "beat": v[4], "synth": v[5],
                            "fx": v[6], "melody": v[7], "vocal": v[8], "drop": v[9]} for k, v in engine.PRESETS.items()}}

@app.post("/api/jobs")
async def create(file: UploadFile = File(...), params: str = Form("{}"), job_id: str = Form("")):
    p = json.loads(params)
    if job_id and job_id in JOBS: jid = job_id; d = WORK / jid  # "gerar nova versão": reaproveita o upload
    else:
        jid = uuid.uuid4().hex; d = WORK / jid; d.mkdir(); await save_upload(file, d)
    job = JOBS.setdefault(jid, {}); job.update(t=time.time(), progress=0.0, msg="Na fila", status="running", error=None)
    seed = int(p.get("seed", 0))
    def run():
        def prog(v, m): job.update(progress=v, msg=m)
        try:
            out = d / "remix.wav"
            info = engine.process(str(d / "original.wav"), str(out), p, prog, seed)
            if p.get("use_generative") and generative.available():
                prog(.9, "Gerando camada por IA generativa")
                g = generative.generate_bed(f"brazilian funk, eletrofunk goiano, {info['target_bpm']} bpm, heavy 808 bass, tamborzao",
                                            str(d / "original.wav"), str(d / "gen.wav"))
                a, _ = sf.read(out); b, _ = sf.read(g, always_2d=True); b = np.resize(b, (len(a), 2)) * .5
                m = a + b; sf.write(out, m / max(1, np.abs(m).max()) * .89, engine.SR)
                info["generative"] = True
            job.update(status="done", info=info)
        except Exception as e:
            job.update(status="error", error=str(e) or e.__class__.__name__)
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    jid = uuid.uuid4().hex; d = WORK / jid; d.mkdir(); JOBS[jid] = {"t": time.time()}
    await save_upload(file, d)
    try: return {"job_id": jid, **engine.analyze(str(d / "original.wav"))}
    except ValueError as e: raise HTTPException(400, str(e))

@app.get("/api/jobs/{jid}")
def status(jid: str):
    j = JOBS.get(jid)
    if not j: raise HTTPException(404, "Trabalho não encontrado ou expirado. Envie o arquivo novamente.")
    return {k: j.get(k) for k in ("status", "progress", "msg", "error", "info")}

@app.get("/api/jobs/{jid}/audio/{which}")
def audio(jid: str, which: str, fmt: str = "wav", dl: int = 0):
    if which not in ("original", "remix") or fmt not in ("wav", "mp3") or jid not in JOBS: raise HTTPException(404)
    wav = WORK / jid / f"{which}.wav"
    if not wav.exists(): raise HTTPException(404, "Áudio ainda não disponível.")
    f = wav
    if fmt == "mp3":
        f = wav.with_suffix(".mp3")
        if not f.exists():
            try: engine.export_mp3(str(wav), str(f))
            except RuntimeError as e: raise HTTPException(500, str(e))
    return FileResponse(f, filename=f"jf-eletrofunk-{which}.{fmt}" if dl else None)

@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    shutil.rmtree(WORK / jid, ignore_errors=True); JOBS.pop(jid, None); return {"ok": True}

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(HERE / "index.html")
