"""Gancho OPCIONAL de IA generativa (MusicGen via Replicate).
Desligado por padrão. Ative com a variável de ambiente REPLICATE_API_TOKEN (serviço PAGO, exige internet).
Sem o token, o app usa só o motor DSP local e NÃO afirma que há IA generativa.
NÃO TESTADO (o ambiente de desenvolvimento não tinha rede): valide com sua chave antes de publicar."""
import os, urllib.request

def available():
    return bool(os.environ.get("REPLICATE_API_TOKEN"))

def generate_bed(prompt, melody_wav, out_wav, seconds=30):
    import replicate  # pip install replicate
    model = os.environ.get("GEN_MODEL", "meta/musicgen")
    with open(melody_wav, "rb") as f:
        out = replicate.run(model, input={
            "prompt": prompt, "input_audio": f, "duration": min(int(seconds), 30),
            "model_version": "stereo-melody-large", "output_format": "wav"})
    urllib.request.urlretrieve(str(out), out_wav)
    return out_wav
