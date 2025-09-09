import streamlit as st
import tempfile, os, time, uuid
import cv2
import numpy as np
import torch

# -------------------------------
# Clear cache
# -------------------------------
st.cache_data.clear()
st.cache_resource.clear()

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Audiowide&display=swap');
body { background-color: #0d1117; color: #c9d1d9; }
h1,h2,h3,h4,h5,h6 { font-family: 'Orbitron', sans-serif; color: #00fff7; }
.stButton>button { background-color:#00fff7; color:#0d1117; border-radius:12px; font-weight:bold; font-family: 'Audiowide', sans-serif; transition: transform 0.2s; }
.stButton>button:hover { transform: scale(1.05); }
.stSlider>div { background: linear-gradient(90deg, #00f7ff, #ff00ff); }
.stProgress>div>div { background: linear-gradient(90deg, #00f7ff, #ff00ff); }
.stSelectbox>div>div, .stFileUploader>div>div { background: rgba(255,255,255,0.05); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load MiDaS model
# -------------------------------
@st.cache_resource
def load_midas():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return midas, transform, device

midas, midas_transform, device = load_midas()

# -------------------------------
# Video processing functions
# -------------------------------
def estimate_depth(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy().astype(np.float32)
    depth -= depth.min()
    if depth.max() > 0:
        depth /= depth.max()
    depth = 1.0 - depth
    depth = cv2.GaussianBlur(depth, (7, 7), 0)
    return depth

def make_stereo(frame_bgr, depth_norm, max_disp=24):
    h, w = depth_norm.shape
    disp = (depth_norm * max_disp).astype(np.float32)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x_left = (x + disp).astype(np.float32)
    map_x_right = (x - disp).astype(np.float32)
    left = cv2.remap(frame_bgr, map_x_left, y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    right = cv2.remap(frame_bgr, map_x_right, y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sbs = cv2.hconcat([left, right])
    return left, right, sbs

def make_anaglyph(left, right):
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    anaglyph = np.zeros_like(left)
    anaglyph[..., 0] = left_gray
    anaglyph[..., 1] = right_gray
    anaglyph[..., 2] = right_gray
    return anaglyph

def process_video(input_path, max_disp=24, target_height=480, every_n=1, progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = target_height / height
    new_w = int(width * scale)
    new_h = target_height

    # ensure even dimensions
    new_w += new_w % 2
    new_h += new_h % 2
    sbs_w = new_w * 2

    out_fps = max(1.0, fps / every_n)

    # temporary files
    out_sbs_path = os.path.join(tempfile.gettempdir(), f"sbs_{uuid.uuid4().hex}.mp4")
    out_ana_path = os.path.join(tempfile.gettempdir(), f"ana_{uuid.uuid4().hex}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264, browser-friendly
    writer_sbs = cv2.VideoWriter(out_sbs_path, fourcc, out_fps, (sbs_w, new_h))
    writer_ana = cv2.VideoWriter(out_ana_path, fourcc, out_fps, (new_w, new_h))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n != 0:
            frame_idx += 1
            continue
        frame = cv2.resize(frame, (new_w, new_h))
        depth = estimate_depth(frame)
        left, right, sbs = make_stereo(frame, depth, max_disp=max_disp)
        anaglyph = make_anaglyph(left, right)

        writer_sbs.write(sbs)
        writer_ana.write(anaglyph)
        frame_idx += 1

        if progress_callback:
            progress_callback(min(int(frame_idx / total_frames * 100), 100))

    cap.release()
    writer_sbs.release()
    writer_ana.release()

    return out_sbs_path, out_ana_path, total_frames, frame_idx, out_fps

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Dream2VR — 2D ➜ VR", page_icon="🎬", layout="centered")
st.title("🎬 Dream2VR — 2D ➜ VR")
st.caption("Upload a 2D video and preview SBS & Anaglyph 3D videos.")

uploaded = st.file_uploader("Upload a short MP4 (5–15s is best)", type=["mp4", "mov", "m4v"])
if uploaded:
    st.video(uploaded)

st.subheader("⚙️ Settings")
max_disp = st.slider("Max disparity (3D effect strength)", 4, 64, 24)
target_h = st.selectbox("Target height", [360, 480, 720], index=1)
every_n = st.selectbox("Process every Nth frame (speed boost)", [1, 2, 3], index=0)

process_btn = st.button("🚀 Process Video")

if process_btn and uploaded:
    with st.spinner("🤖 Processing video frames..."):
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_in.write(uploaded.read())
        temp_in.close()

        progress_bar = st.progress(0)

        def update_progress(pct):
            progress_bar.progress(pct)

        t0 = time.time()
        sbs_path, ana_path, total_frames, written_frames, out_fps = process_video(
            temp_in.name,
            max_disp=max_disp,
            target_height=target_h,
            every_n=every_n,
            progress_callback=update_progress
        )
        dt = time.time() - t0

        st.success(f"✅ Done in {dt:.1f} seconds")
        st.info(f"🎥 Frames processed: {written_frames} / {total_frames}, Output FPS: {out_fps:.2f}")

        # -------------------------------
        # SBS Video
        # -------------------------------
        st.subheader("📺 SBS VR Video")
        with open(sbs_path, "rb") as f:
            sbs_bytes = f.read()
        st.video(sbs_bytes)
        st.download_button("⬇️ Download SBS Video", sbs_bytes, file_name="dream2vr_sbs.mp4")

        # -------------------------------
        # Anaglyph Video
        # -------------------------------
        st.subheader("👓 Anaglyph 3D Video (Red/Cyan Glasses)")
        with open(ana_path, "rb") as f:
            ana_bytes = f.read()
        st.video(ana_bytes)
        st.download_button("⬇️ Download Anaglyph Video", ana_bytes, file_name="dream2vr_anaglyph.mp4")
