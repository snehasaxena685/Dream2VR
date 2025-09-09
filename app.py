import streamlit as st
import streamlit.components.v1 as components
import tempfile, os, time, uuid
import cv2
import numpy as np
import torch
import base64

# -------------------------------
# Clear cache to avoid old function issues
# -------------------------------
st.cache_data.clear()
st.cache_resource.clear()

# -------------------------------
# Custom CSS for futuristic look
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Audiowide&display=swap');

body {
    background-color: #0d1117;
    color: #c9d1d9;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif;
    color: #00fff7;
}

.stButton>button {
    background-color:#00fff7;
    color:#0d1117;
    border-radius:12px;
    font-weight:bold;
    font-family: 'Audiowide', sans-serif;
    transition: transform 0.2s;
}
.stButton>button:hover { transform: scale(1.05); }

.stSlider>div { background: linear-gradient(90deg, #00f7ff, #ff00ff); }

.stProgress>div>div {
    background: linear-gradient(90deg, #00f7ff, #ff00ff);
}

.stSelectbox>div>div, .stFileUploader>div>div {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load API Key (future use)
# -------------------------------
API_KEY = st.secrets.get("API_KEY", os.getenv("API_KEY", None))
if API_KEY:
    st.sidebar.success("🔑 API key loaded")
else:
    st.sidebar.warning("⚠️ No API key found. Local processing only.")

# -------------------------------
# Load MiDaS depth model (cached)
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
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    map_x_left = x + disp
    map_x_right = x - disp
    left = cv2.remap(frame_bgr, map_x_left, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    right = cv2.remap(frame_bgr, map_x_right, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
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

def process_video(input_path, output_path, max_disp=24, target_height=480, every_n=1, progress_callback=None):
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
    out_fps = max(1.0, fps / every_n)
    
    # -------------------------------
    # Use portable codec for Streamlit Cloud
    # -------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width_even = (new_w * 2) // 2 * 2  # ensure even width
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width_even, new_h))
    if not writer.isOpened():
        raise RuntimeError("⚠️ VideoWriter failed to open. Try another codec.")

    anaglyph_preview = None
    frame_idx = 0
    written_frames = 0

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
        writer.write(sbs)
        written_frames += 1
        if anaglyph_preview is None:
            anaglyph_preview = make_anaglyph(left, right)
        if progress_callback:
            percent = int((frame_idx / total_frames) * 100)
            progress_callback(min(percent, 100))
        frame_idx += 1

    cap.release()
    writer.release()

    if written_frames == 0:
        raise RuntimeError("⚠️ No frames were written to output video.")

    return output_path, anaglyph_preview, total_frames, written_frames, out_fps

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Dream2VR — 2D ➜ VR", page_icon="🎬", layout="centered")
st.title("🎬 Dream2VR — 2D ➜ VR")
st.caption("Upload a 2D video and preview it in VR, SBS, or Anaglyph 3D mode.")

uploaded = st.file_uploader("Upload a short MP4 (5–15s is best)", type=["mp4", "mov", "m4v"])
if uploaded:
    st.video(uploaded)

st.subheader("⚙️ Settings")
with st.container():
    max_disp = st.slider("Max disparity (3D effect strength)", 4, 64, 24)
    target_h = st.selectbox("Target height", [360, 480, 720], index=1)
    every_n = st.selectbox("Process every Nth frame (speed boost)", [1, 2, 3], index=0)
    preview_modes = st.multiselect(
        "Select preview modes:",
        ["SBS", "WebVR", "Anaglyph"],
        default=["SBS", "WebVR"]
    )

process_btn = st.button("🚀 Process Video", type="primary", use_container_width=True)

if process_btn and uploaded:
    with st.spinner("🤖 AI processing video frames..."):
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_in.write(uploaded.read())
        temp_in.close()
        out_path = os.path.join(tempfile.gettempdir(), f"out_{uuid.uuid4().hex}.mp4")

    progress_bar = st.progress(0, text="Initializing AI...")

    def update_progress(pct):
        progress_bar.progress(pct, text=f"Processing... {pct}%")

    t0 = time.time()
    
    # -------------------------------
    # Safety check for unpacking
    # -------------------------------
    result = process_video(
        temp_in.name,
        out_path,
        max_disp=max_disp,
        target_height=target_h,
        every_n=every_n,
        progress_callback=update_progress
    )

    if isinstance(result, tuple) and len(result) == 5:
        final_out, anaglyph_frame, total_frames, written_frames, out_fps = result
    else:
        st.error("❌ process_video did not return expected 5 values")
        st.stop()

    dt = time.time() - t0

    st.success(f"✅ Done in {dt:.1f} seconds")
    st.info(f"🎥 Input frames: {total_frames} | Written frames: {written_frames} | Output FPS: {out_fps:.2f}")

    with open(final_out, "rb") as f:
        video_bytes = f.read()
        b64_video = base64.b64encode(video_bytes).decode()

        if "SBS" in preview_modes:
            st.subheader("📺 SBS VR Video")
            st.video(video_bytes)
            st.download_button("⬇️ Download SBS VR Video", video_bytes, file_name="dream2vr_sbs.mp4")

        if "WebVR" in preview_modes:
            st.subheader("🕶️ Interactive WebVR Preview")
           # Save final video to temp file and get its path
video_url = final_out.replace("\\", "/")  # make sure path separators are correct

aframe_html = f"""
<html>
<head>
  <script src="https://aframe.io/releases/1.5.0/aframe.min.js"></script>
</head>
<body style="margin:0; background:black;">
  <a-scene>
    <a-videosphere src="file://{video_url}" autoplay="true" loop="true" rotation="0 -90 0"></a-videosphere>
    <a-camera wasd-controls-enabled="true" look-controls="true"></a-camera>
  </a-scene>
</body>
</html>
"""
components.html(aframe_html, height=500)
if "Anaglyph" in preview_modes and anaglyph_frame is not None:
            st.subheader("👓 Anaglyph 3D Preview (Red/Cyan Glasses)")
            st.image(anaglyph_frame, channels="BGR", caption="Preview with red/cyan glasses")
