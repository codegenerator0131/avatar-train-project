#!/usr/bin/env bash
# Stage 4: Text → Voice (XTTS v2) → FLAME params (CodeTalker + FLAME fitting)
# Full pipeline: text → cloned speech → mesh animation → per-frame .npz → ready for Stage 5
set -euo pipefail
trap 'echo ""; echo "ERROR: Stage 4 failed at line $LINENO. Command: $BASH_COMMAND"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_DIR="$SCRIPT_DIR/elite"
STAGE4_DIR="$SCRIPT_DIR/stage4"
TTS_PYTHON="$HOME/miniconda3/envs/tts/bin/python"
ELITE_PYTHON="$HOME/miniconda3/envs/ELITE/bin/python"

echo "================================================"
echo "  Stage 4: Text → Speech → FLAME Animation"
echo "================================================"
echo ""

# ── Manual download check ──────────────────────────────────────────────────────
CODETALKER_ST1_CHECK="$SCRIPT_DIR/stage4/CodeTalker/vocaset/vocaset_stage1.pth.tar"
CODETALKER_ST2_CHECK="$SCRIPT_DIR/stage4/CodeTalker/vocaset/vocaset_stage2.pth.tar"
MISSING=0
if [ -f "$CODETALKER_ST1_CHECK" ] && [ "$(stat -c%s "$CODETALKER_ST1_CHECK")" -lt 100000 ]; then rm -f "$CODETALKER_ST1_CHECK"; fi
if [ -f "$CODETALKER_ST2_CHECK" ] && [ "$(stat -c%s "$CODETALKER_ST2_CHECK")" -lt 100000 ]; then rm -f "$CODETALKER_ST2_CHECK"; fi
if [ ! -f "$CODETALKER_ST1_CHECK" ] || [ ! -f "$CODETALKER_ST2_CHECK" ]; then
    echo "  ┌─────────────────────────────────────────────────────────┐"
    echo "  │  MANUAL DOWNLOAD REQUIRED (one-time setup)              │"
    echo "  │                                                         │"
    echo "  │  CodeTalker checkpoints must be downloaded from         │"
    echo "  │  Google Drive and placed at:                            │"
    echo "  │                                                         │"
    if [ ! -f "$CODETALKER_ST1_CHECK" ]; then
    echo "  │  • vocaset_stage1.pth.tar                               │"
    echo "  │    https://drive.google.com/file/d/1RszIMsxcWX7WPlaODqJvax8M_dnCIzk5│"
    echo "  │    → stage4/CodeTalker/vocaset/vocaset_stage1.pth.tar  │"
    fi
    if [ ! -f "$CODETALKER_ST2_CHECK" ]; then
    echo "  │  • vocaset_stage2.pth.tar                               │"
    echo "  │    https://drive.google.com/file/d/1phqJ_6AqTJmMdSq-__KY6eVwN4J9iCGP│"
    echo "  │    → stage4/CodeTalker/vocaset/vocaset_stage2.pth.tar  │"
    fi
    echo "  │                                                         │"
    echo "  │  See DOWNLOADS.md for all required files.               │"
    echo "  └─────────────────────────────────────────────────────────┘"
    echo ""
    MISSING=1
fi
if [ "$MISSING" = "1" ]; then
    echo "  Attempting auto-download via gdown..."
    mkdir -p "$SCRIPT_DIR/stage4/CodeTalker/vocaset"
fi

# ── Inputs ────────────────────────────────────────────────────────────────────
DEFAULT_VIDEO=""
for f in "$SCRIPT_DIR/data/capture/IMG_9625.mov" "$SCRIPT_DIR/data/capture/IMG_9625.MOV"; do
    [ -f "$f" ] && DEFAULT_VIDEO="$f" && break
done
read -e -i "${DEFAULT_VIDEO:-$SCRIPT_DIR/data/capture/IMG_9625.mov}" -p "Input video path: " VIDEO
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

ID="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
VOICE_SAMPLE="$SCRIPT_DIR/data/capture/${ID}_voice_sample.wav"

echo ""
read -e -p "Text to speak: " SPEAK_TEXT
if [ -z "$SPEAK_TEXT" ]; then
    echo "ERROR: No text provided"
    exit 1
fi

read -e -i "0" -p "GPU index (or -1 for CPU): " GPU
export CUDA_VISIBLE_DEVICES=$GPU

# Sanitize text for filename
SAFE_TEXT=$(echo "$SPEAK_TEXT" | tr ' ' '_' | tr -cd '[:alnum:]_' | cut -c1-40)
OUTPUT_DIR="$SCRIPT_DIR/data/stage4/${ID}/${SAFE_TEXT}"
mkdir -p "$OUTPUT_DIR"

export CUDA_HOME="${CUDA_HOME:-$HOME/miniconda3/envs/ELITE}"
export PYTHONPATH="$ELITE_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo ""
echo "Person ID: $ID"
echo "Text:      $SPEAK_TEXT"
echo "Output:    $OUTPUT_DIR"
echo ""

# ── Step 1/4 — Extract voice sample ───────────────────────────────────────────
echo "Step 1/4 — Extracting voice sample from video..."
echo ""

AUDIO_FULL="$SCRIPT_DIR/data/capture/${ID}.wav"
if [ -f "$AUDIO_FULL" ]; then
    echo "  [skip] Full audio already extracted: $AUDIO_FULL"
else
    ffmpeg -i "$VIDEO" -vn -acodec pcm_s16le -ar 22050 -ac 1 "$AUDIO_FULL" -y
    echo "  Extracted: $AUDIO_FULL"
fi

if [ -f "$VOICE_SAMPLE" ]; then
    echo "  [skip] Voice sample already exists: $VOICE_SAMPLE"
else
    ffmpeg -i "$AUDIO_FULL" -t 30 -acodec pcm_s16le -ar 22050 -ac 1 "$VOICE_SAMPLE" -y
    echo "  Voice sample (30s): $VOICE_SAMPLE"
fi

# ── Step 2/4 — Generate speech with voice cloning ─────────────────────────────
echo ""
echo "Step 2/4 — Generating speech (XTTS v2 voice cloning)..."
echo ""

OUTPUT_WAV="$OUTPUT_DIR/speech.wav"
OUTPUT_WAV_16K="$OUTPUT_DIR/speech_16k.wav"

if [ -f "$OUTPUT_WAV" ]; then
    echo "  [skip] Speech audio already exists: $OUTPUT_WAV"
else
    # Create tts env if missing
    if [ ! -f "$TTS_PYTHON" ]; then
        echo "  Creating tts conda environment..."
        "$HOME/miniconda3/bin/conda" create -n tts python=3.10 -y
        "$TTS_PYTHON" -m pip install TTS==0.22.0
    fi
    "$TTS_PYTHON" -c "import TTS" 2>/dev/null || "$TTS_PYTHON" -m pip install TTS==0.22.0
    # Pin deps for TTS==0.22.0 compatibility:
    #   transformers<4.40  — BeamSearchScorer removed in 4.40+
    #   torchaudio==2.1.0  — newer torchaudio requires torchcodec which TTS doesn't use
    "$TTS_PYTHON" -m pip install \
        "transformers==4.39.3" \
        "torchaudio==2.1.0" --index-url https://download.pytorch.org/whl/cu121 \
        --quiet
    # PyTorch 2.6+ changed torch.load default to weights_only=True — patch TTS io.py
    TTS_IO="$HOME/miniconda3/envs/tts/lib/python3.10/site-packages/TTS/utils/io.py"
    if [ -f "$TTS_IO" ]; then
        sed -i \
            -e 's/torch\.load(f, map_location=map_location, \*\*kwargs)/torch.load(f, map_location=map_location, weights_only=False, **kwargs)/g' \
            -e 's/torch\.load(f, map_location=map_location)/torch.load(f, map_location=map_location, weights_only=False)/g' \
            "$TTS_IO"
        echo "  Patched TTS io.py for PyTorch 2.6+"
    fi

    USE_GPU="True"
    [ "$GPU" = "-1" ] && USE_GPU="False"

    TTS_HOME="${TTS_HOME:-$HOME/.local/share/tts}" "$TTS_PYTHON" - <<PYEOF
import os
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS
import torch

device = "cuda" if ($USE_GPU and torch.cuda.is_available()) else "cpu"
print(f"  TTS device: {device}")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts.tts_to_file(
    text="$SPEAK_TEXT",
    speaker_wav="$VOICE_SAMPLE",
    language="en",
    file_path="$OUTPUT_WAV",
)
print(f"  Generated: $OUTPUT_WAV")
PYEOF
fi

if [ -f "$OUTPUT_WAV_16K" ]; then
    echo "  [skip] 16kHz audio already exists"
else
    ffmpeg -i "$OUTPUT_WAV" -ar 16000 -ac 1 "$OUTPUT_WAV_16K" -y
    echo "  Converted to 16kHz mono: $OUTPUT_WAV_16K"
fi

# ── Step 3/4 — CodeTalker: audio → mesh vertices ──────────────────────────────
echo ""
echo "Step 3/4 — Generating facial animation (CodeTalker VOCASET)..."
echo ""

CODETALKER_DIR="$STAGE4_DIR/CodeTalker"
VERTICES_NPY="$OUTPUT_DIR/vertices.npy"
CODETALKER_ST1="$CODETALKER_DIR/vocaset/vocaset_stage1.pth.tar"
CODETALKER_ST2="$CODETALKER_DIR/vocaset/vocaset_stage2.pth.tar"

if [ -f "$VERTICES_NPY" ]; then
    echo "  [skip] Vertices already generated: $VERTICES_NPY"
else
    # Clone CodeTalker if missing
    if [ ! -d "$CODETALKER_DIR" ]; then
        echo "  Cloning CodeTalker..."
        mkdir -p "$STAGE4_DIR"
        git clone https://github.com/Doubiiu/CodeTalker.git "$CODETALKER_DIR"
    fi

    # Install CodeTalker deps into ELITE env
    echo "  Installing CodeTalker dependencies..."
    "$ELITE_PYTHON" -m pip install librosa==0.9.2 transformers==4.27.4 --quiet 2>/dev/null || true

    # Download VOCASET checkpoints if missing
    mkdir -p "$CODETALKER_DIR/vocaset"


    # Checkpoints are on Google Drive — use gdown
    "$ELITE_PYTHON" -m pip install "gdown>=4.6.0" --quiet
    gdrive_download() {
        local file_id="$1" dst="$2" name="$3"
        if [ -f "$dst" ] && [ "$(stat -c%s "$dst" 2>/dev/null || echo 0)" -lt 100000 ]; then
            echo "  Removing corrupt $name..."; rm -f "$dst"
        fi
        [ -f "$dst" ] && return 0
        echo "  Downloading $name from Google Drive..."
        "$ELITE_PYTHON" -m gdown "https://drive.google.com/uc?id=${file_id}" -O "$dst"
    }
    gdrive_download "1phqJ_6AqTJmMdSq-__KY6eVwN4J9iCGP" "$CODETALKER_ST2" "vocaset_stage2.pth.tar"
    gdrive_download "1RszIMsxcWX7WPlaODqJvax8M_dnCIzk5" "$CODETALKER_ST1" "vocaset_stage1.pth.tar"

    # Generate FLAME neutral template if missing
    FLAME_TEMPLATE_PKL="$CODETALKER_DIR/vocaset/FLAME_template.pkl"
    if [ ! -f "$FLAME_TEMPLATE_PKL" ]; then
        echo "  Generating FLAME neutral template..."
        "$ELITE_PYTHON" - <<PYEOF2
import sys, torch, numpy as np, pickle
sys.path.insert(0, "$ELITE_DIR")
from vhap.model.flame import FlameHead
import yaml

with open("$ELITE_DIR/configs/3d_prior.yaml") as f:
    cfg = yaml.safe_load(f)
fc = cfg['flame']

# Use add_teeth=False so vertex count = 5023 (15069 floats) matching VOCASET checkpoint
flame = FlameHead(fc['n_shape'], fc['n_expr'],
    add_teeth=False, remove_lip_inside=False)
flame.eval()

with torch.no_grad():
    out = flame.forward(
        shape=torch.zeros(1, fc['n_shape']), expr=torch.zeros(1, fc['n_expr']),
        rotation=torch.zeros(1, 3), neck=torch.zeros(1, 3),
        jaw=torch.zeros(1, 3), eyes=torch.zeros(1, 6),
        translation=torch.zeros(1, 3), zero_centered_at_root_node=True,
        return_landmarks=False, return_verts_cano=False,
        static_offset=None, dynamic_offset=None)
    verts = out[0] if isinstance(out, (tuple, list)) else out

template = verts[0].cpu().numpy()
print(f"  FLAME template: {template.shape[0]} vertices")
template_dict = {'flame_neutral': template.reshape(-1)}
with open("$FLAME_TEMPLATE_PKL", 'wb') as f:
    pickle.dump(template_dict, f)
print(f"  Saved: $FLAME_TEMPLATE_PKL")
PYEOF2
    fi

    # Run CodeTalker inference
    "$ELITE_PYTHON" - <<PYEOF3
import sys, os, torch, numpy as np, pickle, librosa
sys.path.insert(0, "$CODETALKER_DIR")
sys.path.insert(0, "$ELITE_DIR")

from transformers import Wav2Vec2Processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  CodeTalker running on: {device}")

# Load audio (raw waveform — CodeTalker processes it internally via Wav2Vec2)
speech_array, _ = librosa.load("$OUTPUT_WAV_16K", sr=16000)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_input = np.squeeze(
    processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
)
# CodeTalker expects shape (1, T_samples)
audio_feature = torch.FloatTensor(audio_input).unsqueeze(0).to(device)
print(f"  Audio shape: {audio_feature.shape}")

# Load FLAME template
with open("$CODETALKER_DIR/vocaset/FLAME_template.pkl", 'rb') as f:
    templates = pickle.load(f, encoding='latin1')
template = torch.FloatTensor(templates['flame_neutral'].reshape(1, -1)).to(device)
print(f"  Template shape: {template.shape}")

# One-hot condition (8 VOCASET training subjects, use index 0)
NUM_TRAIN_SUBJECTS = 8
one_hot = torch.zeros(1, NUM_TRAIN_SUBJECTS).to(device)
one_hot[0, 0] = 1.0

# Load CodeTalker model — build minimal config namespace
# Load config from CodeTalker's own yaml — flattens all sections into one namespace
import yaml as _yaml
with open("$CODETALKER_DIR/config/vocaset/demo.yaml") as _f:
    _raw = _yaml.safe_load(_f)

class CFG:
    pass

_cfg = CFG()
for _section in _raw.values():
    if isinstance(_section, dict):
        for _k, _v in _section.items():
            setattr(_cfg, _k, _v)

# Override paths to point to our downloaded checkpoints
_cfg.vqvae_pretrained_path = "$CODETALKER_ST1"
_cfg.model_path = "$CODETALKER_ST2"
_cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

from models.stage2 import CodeTalker
model = CodeTalker(_cfg)
# stage2 checkpoint only contains stage2 weights (not autoencoder)
# autoencoder is already loaded inside CodeTalker.__init__ via vqvae_pretrained_path
# so load stage2 weights with strict=False to skip autoencoder keys
ckpt = torch.load(_cfg.model_path, map_location=device)
state = ckpt.get('state_dict', ckpt)
# Remove autoencoder keys from state_dict — they are already loaded in __init__
state = {k: v for k, v in state.items() if not k.startswith('autoencoder.')}
model.load_state_dict(state, strict=False)
model = model.to(device).eval()
print("  Model loaded")

# Inference
with torch.no_grad():
    prediction = model.predict(audio_feature, template, one_hot)
    pred = prediction.squeeze()  # (T, V*3)

if pred.ndim == 1:
    pred = pred.unsqueeze(0)

T, VD = pred.shape
V = VD // 3
vertices_3d = pred.cpu().numpy().reshape(T, V, 3)

np.save("$VERTICES_NPY", vertices_3d)
print(f"  Vertices saved: {vertices_3d.shape} → $VERTICES_NPY")
print(f"  Duration: {T} frames @ 30fps = {T/30:.1f}s")
PYEOF3
fi

# ── Step 4/4 — FLAME fitting: vertices → per-frame .npz ───────────────────────
echo ""
echo "Step 4/4 — Fitting FLAME params from mesh vertices..."
echo ""

FLAME_PARAMS_DIR="$OUTPUT_DIR/flame_param"
TRANSFORMS_JSON="$OUTPUT_DIR/transforms.json"

PARAM_COUNT=0
if [ -d "$FLAME_PARAMS_DIR" ]; then
    PARAM_COUNT=$(find "$FLAME_PARAMS_DIR" -name "*.npz" 2>/dev/null | wc -l || echo "0")
fi

if [ -f "$TRANSFORMS_JSON" ] && [ "$PARAM_COUNT" -gt 0 ]; then
    echo "  [skip] FLAME params already exist ($PARAM_COUNT frames)"
else
    mkdir -p "$FLAME_PARAMS_DIR"

    "$ELITE_PYTHON" - <<PYEOF4
import sys, os, torch, numpy as np, json
sys.path.insert(0, "$ELITE_DIR")

from vhap.model.flame import FlameHead
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  FLAME fitting on: {device}")

with open("$ELITE_DIR/configs/3d_prior.yaml") as f:
    cfg = yaml.safe_load(f)
fc = cfg['flame']

flame = FlameHead(fc['n_shape'], fc['n_expr'],
    add_teeth=fc['add_teeth'], remove_lip_inside=fc['remove_lip_inside'],
    face_clusters=("skin","hair","boundary","lips_tight","teeth","sclerae","irises")
).to(device)
flame.eval()

# Load CodeTalker vertices
vertices = np.load("$VERTICES_NPY")  # (T, V, 3)
T, V, _ = vertices.shape
print(f"  Fitting {T} frames ({V} vertices)...")

# Load person identity from Stage 2/3 tracked params
tracked_base = "$ELITE_DIR/data/source/tracked/${ID}_whiteBg_staticOffset"
if not os.path.isdir(tracked_base):
    raise FileNotFoundError(
        f"Tracked params not found: {tracked_base}\n"
        "Run Stage 2 + Stage 3 before Stage 4."
    )
last_run = sorted(os.listdir(tracked_base))[-1]
pid = np.load(os.path.join(tracked_base, last_run, 'tracked_flame_params_30.npz'), allow_pickle=True)

pid_shape = torch.FloatTensor(pid['shape'][0:1]).to(device)         # (1, n_shape)
pid_stoffset = torch.FloatTensor(pid['static_offset'][0:1]).to(device)  # (1, V, 3)
pid_rot   = pid['rotation'][0]    # neutral head orientation (3,)
pid_trans = pid['translation'][0] # head position (3,)
pid_neck  = pid['neck_pose'][0]   # neck (3,)
pid_eyes  = pid['eyes_pose'][0]   # eyes (6,)

rot_t   = torch.FloatTensor(pid_rot).unsqueeze(0).to(device)
trans_t = torch.FloatTensor(pid_trans).unsqueeze(0).to(device)
neck_t  = torch.FloatTensor(pid_neck).unsqueeze(0).to(device)
eyes_t  = torch.FloatTensor(pid_eyes).unsqueeze(0).to(device)

# Gradient descent to fit expr + jaw per frame
os.makedirs("$FLAME_PARAMS_DIR", exist_ok=True)
for t in range(T):
    target = torch.FloatTensor(vertices[t]).unsqueeze(0).to(device)  # (1, V, 3)

    expr = torch.zeros(1, fc['n_expr'], device=device, requires_grad=True)
    jaw  = torch.zeros(1, 3, device=device, requires_grad=True)
    opt  = torch.optim.Adam([expr, jaw], lr=0.05)

    for step in range(100):
        opt.zero_grad()
        with torch.enable_grad():
            _out = flame.forward(
                shape=pid_shape, expr=expr, rotation=rot_t,
                neck=neck_t, jaw=jaw, eyes=eyes_t, translation=trans_t,
                zero_centered_at_root_node=True,
                return_landmarks=False, return_verts_cano=False,
                static_offset=pid_stoffset, dynamic_offset=None,
            )
            vp = _out[0] if isinstance(_out, (tuple, list)) else _out
            n = min(vp.shape[1], target.shape[1])
            loss = ((vp[:, :n] - target[:, :n]) ** 2).mean()
            loss.backward()
            opt.step()

    np.savez(f"$FLAME_PARAMS_DIR/{t:05d}.npz",
        shape        = pid['shape'][0:1],
        rotation     = pid_rot.reshape(1, -1),
        translation  = pid_trans.reshape(1, -1),
        neck_pose    = pid_neck.reshape(1, -1),
        jaw_pose     = jaw.detach().cpu().numpy().reshape(1, -1),
        eyes_pose    = pid_eyes.reshape(1, -1),
        expr         = expr.detach().cpu().numpy().reshape(1, -1),
        static_offset= pid['static_offset'][0:1],
    )

    if (t + 1) % 30 == 0 or t == T - 1:
        print(f"  Fitted {t+1}/{T}  loss={loss.item():.6f}")

# Build transforms.json for Stage 5
proc_tf_path = ("$ELITE_DIR/data/source/processed"
                "/${ID}_whiteBg_staticOffset_maskBelowLine/transforms.json")
with open(proc_tf_path) as f:
    proc_tf = json.load(f)

base_frame = proc_tf['frames'][0]
frames_out = []
for t in range(T):
    entry = {k: v for k, v in base_frame.items()}
    entry['timestep_index']   = t
    entry['timestep_id']      = f"f{t:06d}"
    entry['camera_id']        = 'our_cam'
    entry['file_path']        = f"images/{t:05d}_00.png"
    entry['fg_mask_path']     = f"fg_masks/{t:05d}_00.png"
    entry['flame_param_path'] = f"flame_param/{t:05d}.npz"
    frames_out.append(entry)

tf_out = {k: v for k, v in proc_tf.items() if k != 'frames'}
tf_out['frames'] = frames_out
with open("$TRANSFORMS_JSON", 'w') as f:
    json.dump(tf_out, f, indent=2)

print(f"  Saved transforms.json ({T} frames) → $TRANSFORMS_JSON")
print(f"  Stage 4 complete! Ready for Stage 5.")
PYEOF4
fi

echo ""
echo "================================================"
echo "  Stage 4 complete!"
echo "================================================"
echo ""
echo "Person:     $ID"
echo "Text:       $SPEAK_TEXT"
echo "Audio:      $OUTPUT_WAV"
echo "FLAME:      $FLAME_PARAMS_DIR"
echo "Transforms: $TRANSFORMS_JSON"
echo ""
echo "Next: run Stage 5, point motion dir to:"
echo "  $OUTPUT_DIR"
