"""
Evidence Generation Service.
Creates visualizations and artifacts for analysis results.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import io
import base64

try:
    import numpy as np
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


class EvidenceGenerator:
    """Generates visual evidence artifacts for deepfake analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_heatmap(
        self,
        frame: np.ndarray,
        attention_weights: np.ndarray,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate Grad-CAM style heatmap overlay.
        
        Args:
            frame: Original frame (H, W, 3)
            attention_weights: Attention map (H', W')
            output_path: Optional output file path
        
        Returns:
            Path to saved heatmap image
        """
        if not CV2_AVAILABLE:
            return ""
        
        # Resize attention to match frame
        h, w = frame.shape[:2]
        heatmap = cv2.resize(attention_weights, (w, h))
        
        # Normalize to 0-255
        heatmap = np.uint8(255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        # Save
        if output_path is None:
            output_path = str(self.output_dir / f"heatmap_{id(frame)}.jpg")
        
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return output_path
    
    def generate_spectrogram(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        output_path: Optional[str] = None,
        highlight_regions: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        """
        Generate audio spectrogram visualization.
        
        Args:
            audio_data: Audio waveform
            sample_rate: Sample rate in Hz
            output_path: Optional output file path
            highlight_regions: List of (start_sec, end_sec) to highlight
        
        Returns:
            Path to saved spectrogram image
        """
        if not PLT_AVAILABLE:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Compute spectrogram
        from scipy import signal
        frequencies, times, spectrogram = signal.spectrogram(
            audio_data, sample_rate, nperseg=1024
        )
        
        # Plot
        im = ax.pcolormesh(
            times, frequencies, 10 * np.log10(spectrogram + 1e-10),
            shading='gouraud', cmap='magma'
        )
        
        # Highlight suspicious regions
        if highlight_regions:
            for start, end in highlight_regions:
                ax.axvspan(start, end, alpha=0.3, color='red')
        
        ax.set_ylabel('Frequency (Hz)', color='white')
        ax.set_xlabel('Time (sec)', color='white')
        ax.tick_params(colors='white')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Save
        if output_path is None:
            output_path = str(self.output_dir / f"spectrogram_{id(audio_data)}.png")
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#1a1a2e')
        plt.close()
        
        return output_path
    
    def generate_timeline(
        self,
        segments: List[Dict[str, Any]],
        duration_ms: int,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate visual timeline of flagged segments.
        
        Args:
            segments: List of segment dicts with start_ms, end_ms, type, score
            duration_ms: Total duration in milliseconds
            output_path: Optional output file path
        
        Returns:
            Path to saved timeline image
        """
        if not PLT_AVAILABLE:
            return ""
        
        fig, ax = plt.subplots(figsize=(14, 2), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        duration_sec = duration_ms / 1000
        
        # Draw base timeline
        ax.axhline(y=0.5, color='#333', linewidth=8, solid_capstyle='round')
        
        # Color mapping for segment types
        type_colors = {
            'video': '#ff6b6b',
            'audio': '#4ecdc4',
            'lipsync': '#ffe66d',
        }
        
        # Draw segments
        for seg in segments:
            start = seg['start_ms'] / 1000
            end = seg['end_ms'] / 1000
            seg_type = seg.get('segment_type', 'video')
            score = seg.get('score', 0.5)
            
            color = type_colors.get(seg_type, '#ff6b6b')
            alpha = 0.5 + score * 0.5
            
            ax.axvspan(start, end, ymin=0.3, ymax=0.7, alpha=alpha, color=color)
            ax.plot([start, start], [0.2, 0.8], color=color, linewidth=2)
        
        # Styling
        ax.set_xlim(0, duration_sec)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (seconds)', color='white')
        ax.set_yticks([])
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, alpha=0.7, label=t.title())
            for t, c in type_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        
        # Save
        if output_path is None:
            output_path = str(self.output_dir / f"timeline_{id(segments)}.png")
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#1a1a2e')
        plt.close()
        
        return output_path
    
    def generate_score_chart(
        self,
        scores: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate radar chart of modality scores.
        
        Args:
            scores: Dict mapping modality names to scores
            output_path: Optional output file path
        
        Returns:
            Path to saved chart image
        """
        if not PLT_AVAILABLE:
            return ""
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Close the radar chart
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color='#ff6b6b')
        ax.fill(angles, values, alpha=0.25, color='#ff6b6b')
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='white', size=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='#888', size=8)
        ax.grid(color='#333', linestyle='--', alpha=0.5)
        
        # Save
        if output_path is None:
            output_path = str(self.output_dir / f"score_chart_{id(scores)}.png")
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#1a1a2e')
        plt.close()
        
        return output_path
    
    def generate_mouth_strip(
        self,
        frames: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate horizontal strip of mouth ROIs over time.
        
        Args:
            frames: List of frame dicts with mouth_roi images
            output_path: Optional output file path
        
        Returns:
            Path to saved strip image
        """
        if not CV2_AVAILABLE:
            return ""
        
        mouth_rois = [f.get('mouth_roi') for f in frames if f.get('mouth_roi') is not None]
        
        if not mouth_rois:
            return ""
        
        # Resize all to same height
        target_height = 50
        resized = []
        for roi in mouth_rois[:30]:  # Limit to 30 frames
            h, w = roi.shape[:2]
            new_w = int(w * target_height / h)
            resized.append(cv2.resize(roi, (new_w, target_height)))
        
        # Concatenate horizontally
        strip = np.hstack(resized)
        
        # Save
        if output_path is None:
            output_path = str(self.output_dir / f"mouth_strip_{id(frames)}.jpg")
        
        cv2.imwrite(output_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        return output_path
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        path = Path(image_path)
        if not path.exists():
            return ""
        
        with open(path, 'rb') as f:
            data = f.read()
        
        ext = path.suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
        }.get(ext, 'image/png')
        
        b64 = base64.b64encode(data).decode('utf-8')
        return f"data:{mime_type};base64,{b64}"
