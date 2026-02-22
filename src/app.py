"""
app.py (Streamlit)

A simple UI wrapper around crowd_analyzer.py:
- Upload a video
- Configure confidence
- Run analysis
- Preview/download outputs

We run the analyzer as a subprocess so the app remains robust and we can capture logs.
"""

import os
import time
from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st


# -----------------------------
# Page configuration + styling
# -----------------------------
st.set_page_config(
    page_title="Crowd Density Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
      .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px 18px;
      }
      .stButton>button {
        border-radius: 12px;
        padding: 0.65rem 1.0rem;
        font-weight: 700;
      }
      .muted { opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs() -> None:
    Path("videos").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_")).strip()
    return cleaned or "video"


def run_analyzer(video_path: str, conf: float) -> tuple[int, str, str]:
    """
    Run the analyzer script with environment overrides and capture stdout/stderr.
    """
    env = os.environ.copy()
    env["VIDEO_PATH_OVERRIDE"] = video_path
    env["CONF_OVERRIDE"] = str(conf)

    result = subprocess.run(
        ["python", "-u", "src/crowd_analyzer.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


ensure_dirs()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("Crowd Density Analyzer")
    st.markdown('<div class="muted">Upload a video and generate crowd analytics.</div>', unsafe_allow_html=True)

    st.markdown("---")
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    st.markdown('<div class="muted">Higher reduces false positives but may miss people.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("Outputs")
    st.write("- Annotated video (MP4)")
    st.write("- CSV counts over time")
    st.write("- Heatmap image")
    st.write("- Summary report")


# -----------------------------
# Main UI
# -----------------------------
st.markdown("## Crowd Density Analyzer")
st.write("Upload a video, run detection, and download the results.")

col1, col2 = st.columns([1.3, 1.0], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Run")
    run_btn = st.button("Analyze", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Notes")
    st.write("Recommended input characteristics:")
    st.write("- Stable camera (minimal shake)")
    st.write("- Good lighting")
    st.write("- Wide view containing multiple people")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Run pipeline
# -----------------------------
if run_btn:
    if uploaded is None:
        st.error("Please upload a video first.")
        st.stop()

    stem = safe_stem(uploaded.name)
    suffix = Path(uploaded.name).suffix.lower()
    saved_path = f"videos/{stem}{suffix}"

    with open(saved_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"Saved video to {saved_path}")

    st.markdown("---")
    st.subheader("Processing")

    progress = st.progress(0)
    status = st.empty()

    status.info("Running analysis...")
    progress.progress(5)

    start = time.time()
    code, out, err = run_analyzer(saved_path, conf)
    elapsed = time.time() - start

    progress.progress(95)

    with st.expander("Logs", expanded=False):
        if out.strip():
            st.code(out)
        if err.strip():
            st.code(err)

    if code != 0:
        status.error("Analysis failed. Review logs for details.")
        progress.progress(100)
        st.stop()

    status.success(f"Complete in {elapsed:.1f} seconds")
    progress.progress(100)

    # Standard output names written by crowd_analyzer.py
    annotated_path = "outputs/annotated.mp4"
    csv_path = "outputs/crowd_data.csv"
    heatmap_path = "outputs/heatmap.png"
    summary_path = "outputs/summary.txt"

    st.markdown("---")
    st.subheader("Results")

    r1, r2 = st.columns([1.2, 1.0], gap="large")

    with r1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Annotated video")
        if os.path.exists(annotated_path):
            st.video(annotated_path)
            with open(annotated_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    f,
                    file_name=f"{stem}_annotated.mp4",
                    use_container_width=True,
                )
        else:
            st.warning("annotated.mp4 not found in outputs/")
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Summary")
        if os.path.exists(summary_path):
            st.text(open(summary_path, "r", encoding="utf-8").read())
        else:
            st.warning("summary.txt not found in outputs/")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Heatmap")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, use_container_width=True)
            with open(heatmap_path, "rb") as f:
                st.download_button(
                    "Download heatmap",
                    f,
                    file_name=f"{stem}_heatmap.png",
                    use_container_width=True,
                )
        else:
            st.warning("heatmap.png not found in outputs/")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Crowd data")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "time_sec" in df.columns:
            chart_cols = [c for c in ["people_count", "people_smooth"] if c in df.columns]
            if chart_cols:
                st.line_chart(df.set_index("time_sec")[chart_cols])

        st.dataframe(df.head(50), use_container_width=True)

        with open(csv_path, "rb") as f:
            st.download_button(
                "Download CSV",
                f,
                file_name=f"{stem}_crowd_data.csv",
                use_container_width=True,
            )
    else:
        st.warning("crowd_data.csv not found in outputs/")
    st.markdown("</div>", unsafe_allow_html=True)