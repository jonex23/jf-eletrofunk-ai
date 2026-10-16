# JF ELETROFUNK AI

Envie uma música (que você tem autorização para usar), ajuste os controles e receba uma versão em eletrofunk goiano, com prévia, comparação A/B e exportação em WAV ou MP3.

## Arquivos (estrutura plana, tudo na raiz)
| Arquivo | Função |
|---|---|
| `app.py` | API FastAPI (upload, jobs, progresso, áudio, exclusão) e serve `index.html` |
| `engine.py` | Motor de áudio: análise, ajuste de BPM, síntese e mixagem |
| `generative.py` | Gancho OPCIONAL de IA generativa (MusicGen/Replicate, pago) |
| `index.html` | Interface (Web Audio: equalizador, medidor de graves, player, A/B) |
| `requirements.txt` | Dependências Python |
| `Dockerfile`, `render.yaml` | Publicação no Render (ou qualquer host Docker) |
| `.replit`, `replit.nix` | Execução no Replit (instala ffmpeg) |

## O que o motor faz
Processamento de sinal local, **sem IA generativa por padrão**: converte com ffmpeg, estima BPM e tom (librosa), ajusta ao BPM alvo, separa harmônico/percussivo e sintetiza tamborzão, sub-grave no tom, synths, sirene e riser, com estrutura build → pausa → drop, sidechain e limitador (~-1 dBFS). A "preservação de vocal" é uma aproximação por filtro (300–3400 Hz), não separação real de voz.
O resultado é a sua faixa sobre uma base de eletrofunk, não um arranjo novo escrito por IA. Para isso, use a camada opcional abaixo.

## IA generativa (opcional, paga, não testada)
Defina `REPLICATE_API_TOKEN` e adicione `replicate` ao `requirements.txt`. Sem o token a opção nem aparece na interface.

## Publicar no Render (recomendado)
1. Suba estes arquivos para um repositório no GitHub (todos na raiz).
2. Render → New → Blueprint (ou Web Service) → escolha o repositório. O `render.yaml` já define Docker, health check e variáveis.
3. Use um plano com CPU/RAM suficientes: librosa e numba são pesados e o plano gratuito pode falhar ou ser lento. Os arquivos temporários ficam em disco efêmero e são apagados após `JOB_TTL_MIN`.

## Publicar/rodar no Replit
Importe o projeto (ou envie os arquivos), aperte Run. O `replit.nix` instala o ffmpeg. Para publicar, use Deploy com o tipo que mantém o servidor ativo.

## Variáveis
`MAX_UPLOAD_MB` (padrão 50), `JOB_TTL_MIN` (30), `WORK_DIR`, `REPLICATE_API_TOKEN`, `GEN_MODEL`.

## Limites
Processamento em CPU (dezenas de segundos a alguns minutos por faixa), não ilimitado: em produção adicione autenticação ou limite de uso e HTTPS. O estado dos jobs fica em memória, então rode com 1 worker. Use apenas áudio autorizado; o projeto não imita voz nem gravações de artistas específicos.
