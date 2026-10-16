"""Motor de transformação em eletrofunk goiano (DSP local, sem IA generativa por padrão).
Etapas: conversão -> análise (BPM/tom) -> ajuste ao BPM alvo -> separação harmônico/percussivo
-> síntese de bateria (tamborzão), sub-grave, synths, sirene, riser -> sidechain -> master com limitador."""
import os, subprocess
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# Predefinições: (rótulo, estilo, BPM, grave, batida, synth, efeitos, melodia, vocal, drop)
PRESETS = {
 "sertanejo": ("Sertanejo → Eletrofunk Goiano", "goiano", 130, 75, 75, 60, 60, 60, 80, 80),
 "funk": ("Funk → Eletrofunk Pesado", "pesado", 135, 85, 90, 65, 60, 40, 70, 85),
 "trap": ("Trap → Eletrofunk de Grave", "grave", 140, 95, 80, 55, 55, 40, 70, 85),
 "pop": ("Pop → Eletrofunk de Baile", "baile", 128, 70, 75, 60, 60, 65, 85, 75),
 "eletronica": ("Eletrônica → Eletrofunk Acelerado", "acelerado", 150, 75, 85, 80, 75, 55, 50, 90),
 "rock": ("Rock → Eletrofunk Industrial", "industrial", 140, 80, 85, 90, 65, 50, 70, 85),
 "acustica": ("Acústica → Eletrofunk Melódico", "melodico", 125, 60, 60, 45, 50, 85, 90, 65),
 "auto": ("Outros → Conversão automática", "auto", 0, 75, 75, 65, 60, 60, 75, 80),
}
# passos de 1/16 por compasso
KICKS = {"goiano": [0,3,6,10,12], "pesado": [0,3,6,8,10,12,14], "grave": [0,3,10,12], "baile": [0,3,6,10,12],
         "acelerado": [0,3,6,8,11,14], "industrial": [0,3,6,10,12,14], "melodico": [0,6,10,12]}
TOMS = [3, 6, 11, 14]

def to_wav(src, dst):
    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "2", "-ar", str(SR), "-sample_fmt", "s16", dst],
                       capture_output=True)
    if r.returncode: raise ValueError("Não foi possível ler este arquivo de áudio. Use MP3, WAV, FLAC ou M4A válidos.")

def load(path):
    y, _ = sf.read(path, always_2d=True, dtype="float32")
    return y.T  # (2, n)

def analyze(path):
    import librosa
    y = load(path).mean(0)
    if len(y) < SR * 5: raise ValueError("O áudio precisa ter pelo menos 5 segundos.")
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR).mean(1)
    best = max(((np.corrcoef(chroma, np.roll(p, k))[0, 1], k, m) for m, p in (("maior", MAJ), ("menor", MIN)) for k in range(12)))
    return {"bpm": round(float(np.atleast_1d(tempo)[0]), 1), "key": NOTES[best[1]], "mode": best[2],
            "root": best[1], "duration": round(len(y) / SR, 1)}

# ---------- sons sintetizados ----------
def _t(d): return np.arange(int(SR * d)) / SR
def kick(g):
    t = _t(.4); f = 45 + 110 * np.exp(-t * 30); ph = 2 * np.pi * np.cumsum(f) / SR
    return g * (np.sin(ph) * np.exp(-t * 7) + .3 * np.random.randn(len(t)) * np.exp(-t * 200))
def tom(g, f0=190):
    t = _t(.18); ph = 2 * np.pi * np.cumsum(f0 * (.6 + .4 * np.exp(-t * 25))) / SR
    return g * np.sin(ph) * np.exp(-t * 22)
def clap(g):
    t = _t(.22); n = np.random.randn(len(t)); n = sosfilt(butter(2, 1200, "hp", fs=SR, output="sos"), n)
    return g * n * (np.exp(-t * 18) + .6 * np.exp(-((t - .012) ** 2) * 9000)) * .5
def hat(g):
    t = _t(.05); return g * sosfilt(butter(2, 7000, "hp", fs=SR, output="sos"), np.random.randn(len(t))) * np.exp(-t * 90) * .4
def stab(freqs, g, d=.14):
    t = _t(d); x = sum(2 * ((f * t) % 1) - 1 for f in freqs) / len(freqs)
    return g * np.tanh(3 * x) * np.exp(-t * 14)
def siren(d, g):
    t = _t(d); f = 900 + 450 * np.sin(2 * np.pi * 1.5 * t); return g * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.2)
def riser(d, g):
    t = _t(d); n = sosfilt(butter(2, [800, 9000], "bp", fs=SR, output="sos"), np.random.randn(len(t)))
    return g * n * (t / d) ** 2 * .6

def put(buf, snd, pos):
    if pos >= len(buf): return
    e = min(len(buf), pos + len(snd)); buf[pos:e] += snd[: e - pos]

def process(src_wav, out_wav, p, progress=lambda v, m: None, seed=0):
    import librosa
    rng = np.random.default_rng(seed); np.random.seed(seed)
    P = lambda k: float(p.get(k, 50)) / 100
    progress(.05, "Analisando andamento e tom")
    info = analyze(src_wav); y = load(src_wav)
    style = p.get("style", "goiano")
    if style == "auto": style = "pesado" if info["bpm"] > 115 else "goiano"
    target = float(np.clip(p.get("bpm", 130), 80, 180))
    ob = min((info["bpm"] * m for m in (.5, 1, 2)), key=lambda b: abs(np.log(target / b)))
    rate = target / ob; warn = None
    if not .7 <= rate <= 1.5: warn = "BPM muito diferente do original: a melodia pode ficar artificial."
    rate = float(np.clip(rate, .7, 1.5))
    progress(.2, "Ajustando ao BPM alvo")
    y = np.stack([librosa.effects.time_stretch(c, rate=rate) for c in y])
    _, beats = librosa.beat.beat_track(y=y.mean(0), sr=SR, start_bpm=target, units="samples")
    if len(beats): y = y[:, int(beats[0]):]
    n = y.shape[1]; step = int(SR * 60 / target / 4); bar = step * 16; nbars = n // bar + 1
    progress(.4, "Separando harmonia, percussão e vocais")
    h = np.stack([librosa.effects.hpss(c, margin=1.5)[0] for c in y])
    hp = sosfilt(butter(4, 110, "hp", fs=SR, output="sos"), h, axis=1)
    mid = sosfilt(butter(4, [300, 3400], "bp", fs=SR, output="sos"), y.mean(0))
    bed = P("melody") * hp + P("vocal") * np.stack([mid, mid]) * .8
    progress(.55, "Gerando bateria, sub-grave e synths")
    D = np.zeros((n + SR, 2)); drums = np.zeros(n + SR); bass = np.zeros(n + SR); syn = np.zeros(n + SR); fx = np.zeros(n + SR); duck = np.zeros(n + SR)
    root = 36 + info["root"]; f0 = 440 * 2 ** ((root - 69) / 12)  # nota grave (~C2..B2)
    third = 2 ** ((3 if info["mode"] == "menor" else 4) / 12)
    chord = [f0 * 4, f0 * 4 * third, f0 * 4 * 1.5]
    kg, bg, sg, eg, dg = .5 + P("beat") * .5, P("bass"), P("synth"), P("fx"), P("drop")
    kicks = KICKS.get(style, KICKS["goiano"])
    for b in range(nbars):
        ph = b % 16; drop = ph >= 8; build = 4 <= ph < 8
        gain = 1.0 if drop else (.6 if build else .0)
        bs = b * bar
        if ph == 8: put(fx, siren(2.0, .35 * eg), bs); put(fx, clap(1.4 * eg), bs)
        if ph in (6, 7): put(fx, riser(60 / target * 4, .5 * eg), bs)
        for s in range(16):
            pos = bs + s * step
            if ph == 7 and s >= 12: continue  # pausa estratégica antes do drop
            if s % 2 == 0 or drop: put(drums, hat(.5 + .5 * (s % 4 == 2)), pos)
            if gain and s in kicks:
                put(drums, kick(kg * gain), pos); duck[pos:pos + int(step * 1.6)] = np.maximum(duck[pos:pos + int(step * 1.6)], 1)
                if drop or style in ("grave", "pesado"):
                    d = int(step * (1.8 if s in (0, 10) else .9)); t = _t(d / SR)
                    put(bass, bg * gain * np.sin(2 * np.pi * f0 * t) * np.minimum(1, t * 200) * np.exp(-t * 4), pos)
            if gain and s in TOMS and (drop or rng.random() < .5): put(drums, tom(.45 * gain, 160 + 40 * rng.random()), pos)
            if gain and s in (4, 12): put(drums, clap(.7 * gain), pos)
            if drop and s in kicks and sg > .05 and s % 2 == 0: put(syn, stab(chord, sg * .35 * dg + .1 * sg), pos)
        if drop and rng.random() < .4 and ph != 15: put(fx, tom(.5, 120), bs + 14 * step)  # virada
    progress(.8, "Mixando e masterizando")
    env = np.convolve(duck, np.ones(int(SR * .02)) / int(SR * .02), "same")[: n]
    bed = bed * (1 - .65 * env[None, :n])
    mix = bed[:, :n].T * .7
    for x, g in ((drums, 1.0), (bass, 1.3), (syn, 1.0), (fx, 1.0)): mix += (x[:n] * g)[:, None]
    mix = np.vstack([mix, np.zeros((0, 2))])
    mix = sosfilt(butter(2, 25, "hp", fs=SR, output="sos"), mix, axis=0)
    mix = np.tanh(mix * (1.2 + .8 * kg)) / np.tanh(1.2 + .8 * kg)      # saturação/limitador suave
    peak = float(np.abs(mix).max()) or 1; mix = mix / peak * .89        # pico em ~-1 dBFS
    sf.write(out_wav, mix.astype("float32"), SR, subtype="PCM_16")
    progress(1.0, "Concluído")
    return {**info, "target_bpm": target, "style": style, "peak_dbfs": -1.0, "warning": warn}

def export_mp3(wav, mp3):
    r = subprocess.run(["ffmpeg", "-y", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "256k", mp3], capture_output=True)
    if r.returncode: raise RuntimeError("Falha ao exportar MP3 (ffmpeg com libmp3lame ausente?).")
