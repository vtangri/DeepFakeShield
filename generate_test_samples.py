#!/usr/bin/env python3
"""
DeepFakeShield Test Media Generator
Generates sample videos for testing DeepFakeShield AI detection pipeline:
1. sample_real_with_audio.mp4 (Authentic visual frames + natural audio track)
2. sample_fake_with_audio.mp4 (Manipulated visual frames + synthetic audio track)
3. sample_real_no_audio.mp4 (Authentic video stream with NO audio track)
4. sample_fake_no_audio.mp4 (Manipulated video stream with NO audio track)
"""

import os
import subprocess
import numpy as np
import cv2

OUTPUT_DIR = "test_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 25
DURATION_SEC = 5
NUM_FRAMES = FPS * DURATION_SEC
WIDTH, HEIGHT = 640, 480

print(f"Generating test videos in ./{OUTPUT_DIR}/...")

def create_video_stream(filename, is_fake=False):
    video_path = os.path.join(OUTPUT_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    for i in range(NUM_FRAMES):
        # Create dark background
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:] = (20, 25, 35) # Dark slate background
        
        # Center coordinates for face simulation
        center_x = WIDTH // 2 + int(np.sin(i / 10.0) * 15)
        center_y = HEIGHT // 2 + int(np.cos(i / 10.0) * 10)
        
        # Face oval base (skin color)
        cv2.ellipse(img, (center_x, center_y), (100, 130), 0, 0, 360, (180, 200, 230), -1)
        
        # Eyes
        eye_y = center_y - 25
        cv2.circle(img, (center_x - 35, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (center_x + 35, eye_y), 12, (255, 255, 255), -1)
        cv2.circle(img, (center_x - 35, eye_y), 5, (50, 30, 20), -1)
        cv2.circle(img, (center_x + 35, eye_y), 5, (50, 30, 20), -1)
        
        # Mouth (animating open/close for lip motion)
        mouth_h = int(10 + 15 * np.abs(np.sin(i / 4.0)))
        cv2.ellipse(img, (center_x, center_y + 40), (25, mouth_h), 0, 0, 360, (50, 40, 150), -1)
        
        if is_fake:
            # Inject deepfake manipulation visual indicators (GAN boundary artifact, noise)
            cv2.rectangle(img, (center_x - 110, center_y - 140), (center_x + 110, center_y + 140), (0, 0, 255), 2)
            noise = np.random.randint(-40, 40, (120, 120, 3), dtype=np.int16)
            roi = img[center_y-60:center_y+60, center_x-60:center_x+60].astype(np.int16) + noise
            img[center_y-60:center_y+60, center_x-60:center_x+60] = np.clip(roi, 0, 255).astype(np.uint8)
            cv2.putText(img, "DEEPFAKE SYNTHESIS TEST", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, "AUTHENTIC VIDEO TEST", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        out.write(img)
        
    out.release()
    return video_path

# 1. Create Raw Video Streams
raw_real_v = create_video_stream("raw_real.mp4", is_fake=False)
raw_fake_v = create_video_stream("raw_fake.mp4", is_fake=True)

# 2. Generate Audio Tracks via FFmpeg
audio_real_path = os.path.join(OUTPUT_DIR, "audio_real.wav")
audio_fake_path = os.path.join(OUTPUT_DIR, "audio_fake.wav")

# Audio 1: 440 Hz Sine Wave (Natural Voice Tone representation)
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION_SEC}",
    "-ar", "16000", "-ac", "1", audio_real_path
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Audio 2: Modulated Chirp Wave (Synthetic Artificial Voice representation)
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=800:duration={DURATION_SEC}",
    "-ar", "16000", "-ac", "1", audio_fake_path
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 3. Combine Video + Audio -> Final Test Clips

# Case A: Real Video WITH Audio
path_real_audio = os.path.join(OUTPUT_DIR, "sample_real_with_audio.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", raw_real_v, "-i", audio_real_path,
    "-c:v", "libx264", "-c:a", "aac", "-shortest", path_real_audio
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Case B: Fake Video WITH Audio
path_fake_audio = os.path.join(OUTPUT_DIR, "sample_fake_with_audio.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", raw_fake_v, "-i", audio_fake_path,
    "-c:v", "libx264", "-c:a", "aac", "-shortest", path_fake_audio
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Case C: Real Video WITHOUT Audio (Explicitly Strip Audio via -an)
path_real_no_audio = os.path.join(OUTPUT_DIR, "sample_real_no_audio.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", raw_real_v, "-an",
    "-c:v", "libx264", path_real_no_audio
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Case D: Fake Video WITHOUT Audio (Explicitly Strip Audio via -an)
path_fake_no_audio = os.path.join(OUTPUT_DIR, "sample_fake_no_audio.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", raw_fake_v, "-an",
    "-c:v", "libx264", path_fake_no_audio
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Clean up temporary raw files
for temp_file in [raw_real_v, raw_fake_v, audio_real_path, audio_fake_path]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print("\n🎉 Test samples generated successfully!")
print("Location:", os.path.abspath(OUTPUT_DIR))
print("Generated Files:")
print(" 1. sample_real_with_audio.mp4  (Real video + Audio track)")
print(" 2. sample_fake_with_audio.mp4  (Deepfake video + Audio track)")
print(" 3. sample_real_no_audio.mp4    (Real video + NO audio track)")
print(" 4. sample_fake_no_audio.mp4    (Deepfake video + NO audio track)")
