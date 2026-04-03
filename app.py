import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import timedelta
import io
import zipfile

st.set_page_config(page_title="VidPrompt AI", page_icon="🎬", layout="wide")

st.title("🎬 VidPrompt AI")
st.caption("**Versi Terbaik** — Adaptive Keyframe Extraction (Pose + Adegan Change)")

# Sidebar pengaturan
st.sidebar.header("⚙️ Pengaturan Algoritma")
pose_threshold = st.sidebar.slider("Pose Change Threshold", 0.15, 0.50, 0.28, 0.01)
min_interval = st.sidebar.slider("Jarak minimal antar keyframe (detik)", 0.5, 2.0, 0.8, 0.1)
target_model = st.sidebar.selectbox("Target AI Video Generator", ["Kling AI", "Runway Gen-4", "Luma Dream Machine", "Veo 2"])

uploaded_file = st.file_uploader("Upload video (disarankan < 3 menit)", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Simpan video sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getbuffer())
        video_path = tmp.name

    if st.button("🚀 Proses Video Sekarang", type="primary", use_container_width=True):
        with st.spinner("Memproses video dengan algoritma terbaik..."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            mp_pose = mp.solutions.pose
            pose_detector = mp_pose.Pose(min_detection_confidence=0.5)
            
            keyframes = []
            timestamps = []
            reasons = []
            prev_pose = None
            prev_frame = None
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = frame_count / fps
                
                # Skip frame untuk kecepatan
                if frame_count % 2 != 0:
                    continue
                
                # Pose Detection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_detector.process(rgb)
                
                pose_changed = False
                if results.pose_landmarks:
                    current_pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
                    if prev_pose is not None:
                        distance = np.linalg.norm(current_pose - prev_pose)
                        if distance > pose_threshold:
                            pose_changed = True
                    prev_pose = current_pose
                
                # Adegan Change (simple motion)
                adegan_changed = False
                if prev_frame is not None:
                    diff = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    motion_score = np.mean(diff)
                    if motion_score > 15:  # threshold motion
                        adegan_changed = True
                prev_frame = frame.copy()
                
                # Simpan keyframe
                if (pose_changed or adegan_changed) and (not timestamps or timestamp - timestamps[-1] > min_interval):
                    reason = "Pose Change" if pose_changed else "Adegan Change"
                    if pose_changed and adegan_changed:
                        reason = "Pose + Adegan Change"
                    
                    keyframes.append(frame.copy())
                    timestamps.append(timestamp)
                    reasons.append(reason)
            
            cap.release()
            os.unlink(video_path)
            
            st.success(f"✅ Selesai! Ditemukan **{len(keyframes)} keyframes**")

            # Tab Results
            tab1, tab2, tab3 = st.tabs(["📸 Keyframe Gallery", "🔥 Prompt Generator", "📤 Download Semua"])

            with tab1:
                st.subheader("Keyframe Gallery")
                cols = st.columns(4)
                for idx, frame in enumerate(keyframes):
                    with cols[idx % 4]:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb, caption=f"{timedelta(seconds=int(timestamps[idx]))}\n{reasons[idx]}", use_column_width=True)

            with tab2:
                st.subheader("Master Prompt")
                prompt = f"""Cinematic 4K video, hyper realistic, extremely detailed, 24fps, film grain.

Keyframe breakdown ({target_model} optimized):
"""
                for i, ts in enumerate(timestamps):
                    prompt += f"- {timedelta(seconds=int(ts))} → {reasons[i]}\n"
                
                prompt += f"""
Style: cinematic lighting, dramatic rim light, smooth camera movement, shallow depth of field, masterpiece, best quality.
Target model: {target_model}"""
                
                st.text_area("Prompt siap copy-paste", prompt, height=300)
                
                st.download_button("📋 Download Prompt TXT", prompt, file_name="vidprompt_prompt.txt")

            with tab3:
                st.subheader("Download semua hasil")
                if st.button("Download ZIP (keyframes + prompt + JSON)"):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        # Prompt
                        zip_file.writestr("prompt.txt", prompt)
                        # JSON metadata
                        metadata = {"keyframes": []}
                        for i, ts in enumerate(timestamps):
                            metadata["keyframes"].append({
                                "timestamp": float(ts),
                                "reason": reasons[i]
                            })
                        zip_file.writestr("metadata.json", str(metadata))
                        # Gambar keyframes
                        for i, frame in enumerate(keyframes):
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(rgb)
                            img_byte = io.BytesIO()
                            img.save(img_byte, format="PNG")
                            zip_file.writestr(f"keyframe_{i:03d}.png", img_byte.getvalue())
                    
                    zip_buffer.seek(0)
                    st.download_button("⬇️ Download ZIP Sekarang", zip_buffer, file_name="VidPrompt_Full_Result.zip", mime="application/zip")

st.info("**Versi Terbaik ini sudah final.** Jalankan di laptop Anda untuk hasil paling cepat.")
