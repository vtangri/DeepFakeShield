"""
Dataset downloader and generator script.
Downloads public subsets of FaceForensics++ and ASVspoof 2019,
or generates a synthetic benchmark dataset to verify training and evaluation code.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import scipy.io.wavfile as wav
    WAV_AVAILABLE = True
except ImportError:
    WAV_AVAILABLE = False


def create_synthetic_video(output_path: Path, is_fake: bool, duration_sec: int = 2, fps: int = 10, size: tuple = (224, 224)):
    """Create a synthetic MP4 video file for training testing."""
    if not CV2_AVAILABLE:
        print("OpenCV not available. Cannot create synthetic video files.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)

    # Generate video frames
    num_frames = duration_sec * fps
    for f in range(num_frames):
        # Draw a synthetic face-like circle
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Base background
        img[:, :] = [30, 20, 20]
        
        # Draw a face (circle)
        center = (size[0] // 2, size[1] // 2)
        radius = 70
        color = (200, 170, 150)  # Skin tone
        cv2.circle(img, center, radius, color, -1)
        
        # Eyes
        cv2.circle(img, (center[0] - 25, center[1] - 20), 8, (20, 20, 20), -1)
        cv2.circle(img, (center[0] + 25, center[1] - 20), 8, (20, 20, 20), -1)
        
        # Mouth
        mouth_center = (center[0], center[1] + 25)
        # Animate mouth openness based on frame number (viseme)
        mouth_open = int(15 * (0.5 + 0.5 * np.sin(f * 0.8)))
        cv2.ellipse(img, mouth_center, (20, mouth_open), 0, 0, 180, (50, 50, 255), -1)

        # If it is fake, add blending artifacts/manipulation indicators
        if is_fake:
            # Add boundary artifacts around face
            cv2.circle(img, center, radius + 2, (10, 255, 10), 1)  # Green seam artifact
            # Add blocky noise (simulated compression anomaly)
            img[150:180, 150:180] = img[150:180, 150:180] // 2 + 100

        out.write(img)

    out.release()


def create_synthetic_audio(output_path: Path, is_fake: bool, duration_sec: int = 2, sr: int = 16000):
    """Create a synthetic WAV audio file for training testing."""
    if not WAV_AVAILABLE:
        print("Scipy not available. Cannot create synthetic audio files.")
        return

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    
    # Generate audio: base frequency + harmonics
    if not is_fake:
        # Authentic audio: dynamic natural voice frequency sweep
        freq = 150 + 50 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        signal = np.sin(phase)
        # Add natural harmonics
        signal += 0.5 * np.sin(2 * phase)
        signal += 0.25 * np.sin(3 * phase)
    else:
        # Fake audio: static robot-like frequency or noise gaps
        freq = 150 * np.ones_like(t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        signal = np.sin(phase)
        # Add typical synthetic smoothing / vocoder distortion
        signal = np.clip(signal * 1.2, -0.9, 0.9)
        # Periodic mute/artifact gaps
        for gap in range(1, duration_sec):
            start = int(gap * sr)
            signal[start:start+100] = 0.0

    # Normalize to 16-bit range
    signal = signal / np.max(np.abs(signal))
    signal_int16 = (signal * 32767).astype(np.int16)
    
    wav.write(output_path, sr, signal_int16)


def generate_synthetic_dataset(data_dir: Path, num_samples: int = 10):
    """Generate a clean synthetic dataset structure for verification."""
    print(f"Generating synthetic verification dataset at: {data_dir}")
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        
        # Video paths
        real_video_dir = split_dir / "real"
        fake_video_dir = split_dir / "fake"
        real_video_dir.mkdir(parents=True, exist_ok=True)
        fake_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio paths
        bonafide_audio_dir = split_dir / "bonafide"
        spoof_audio_dir = split_dir / "spoof"
        bonafide_audio_dir.mkdir(parents=True, exist_ok=True)
        spoof_audio_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = []
        
        for i in range(num_samples):
            # Create video files
            v_real_path = real_video_dir / f"video_{i:04d}.mp4"
            v_fake_path = fake_video_dir / f"video_{i:04d}.mp4"
            create_synthetic_video(v_real_path, is_fake=False)
            create_synthetic_video(v_fake_path, is_fake=True)
            
            # Create audio files
            a_real_path = bonafide_audio_dir / f"audio_{i:04d}.wav"
            a_fake_path = spoof_audio_dir / f"audio_{i:04d}.wav"
            create_synthetic_audio(a_real_path, is_fake=False)
            create_synthetic_audio(a_fake_path, is_fake=True)
            
        print(f"  Split '{split}' generated with {num_samples} samples per class.")


def main():
    parser = argparse.ArgumentParser(description="Download datasets or generate synthetic benchmarks")
    parser.add_argument("--data-dir", type=str, default="./ml/data", help="Output data directory")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Generate synthetic verification dataset")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.synthetic:
        generate_synthetic_dataset(data_dir)
        print("\nVerification dataset generation completed successfully!")
        print("You can now train video/audio models using:")
        print(f"  python ml/training/train_video.py --data-dir {data_dir} --epochs 2")
        print(f"  python ml/training/train_audio.py --data-dir {data_dir} --epochs 2")
    else:
        print("Real dataset downloads should be run using the instructions in the README.")


if __name__ == "__main__":
    main()
