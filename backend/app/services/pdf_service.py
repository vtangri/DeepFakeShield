"""
PDF Report Generation Service using ReportLab.
Generates professional forensic reports for deepfake analysis.
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from app.core.config import settings


# Color palette
COLORS = {
    'primary': colors.HexColor('#6366F1'),      # Indigo
    'success': colors.HexColor('#10B981'),      # Green
    'warning': colors.HexColor('#F59E0B'),      # Amber
    'error': colors.HexColor('#EF4444'),        # Red
    'dark': colors.HexColor('#1E293B'),         # Slate
    'muted': colors.HexColor('#64748B'),        # Slate muted
    'light': colors.HexColor('#F1F5F9'),        # Light gray
}


def get_verdict_color(score: float) -> colors.Color:
    """Get color based on suspicion score."""
    if score < 0.3:
        return COLORS['success']
    elif score < 0.7:
        return COLORS['warning']
    return COLORS['error']


def get_verdict_text(score: float, label: str) -> tuple:
    """Get verdict text and description based on score."""
    if score < 0.3:
        return "LIKELY AUTHENTIC", "No significant manipulation indicators detected."
    elif score < 0.7:
        return "SUSPICIOUS", "Some anomalies detected. Manual review recommended."
    return "HIGHLY LIKELY FAKE", "Strong evidence of manipulation found."


class PDFReportGenerator:
    """Generates professional PDF reports for deepfake analysis."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or settings.STORAGE_PATH) / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            textColor=COLORS['dark'],
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=COLORS['muted'],
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='DSSectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=COLORS['primary'],
            spaceBefore=20,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='DSBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=COLORS['dark'],
            alignment=TA_JUSTIFY,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='DSVerdictText',
            parent=self.styles['Normal'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=10
        ))
    
    def generate_report(
        self,
        job_id: str,
        overall_score: float,
        label: str,
        results: Dict[str, Any] = None,
        summary_text: str = None,
        media_type: str = "video"
    ) -> str:
        """
        Generate a PDF report and return the file path.
        
        Returns:
            str: Path to the generated PDF file
        """
        results = results or {}
        segments = results.get("segments", [])
        
        video = results.get("video", {})
        audio = results.get("audio", {})
        lipsync = results.get("lipsync", {})
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"deepfakeshield_report_{job_id[:8]}_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        story = []
        
        # Header
        story.extend(self._build_header(job_id))
        
        # Verdict Section
        story.extend(self._build_verdict_section(overall_score, label))
        
        # Technical Details Section (New)
        if results.get("technical_summary"):
            story.extend(self._build_technical_section(results))
        
        # Modality Scores & Detailed Analysis
        story.extend(self._build_modality_section(
            video.get("score", 0.0), 
            audio.get("score", 0.0), 
            lipsync.get("score", 0.0), 
            media_type,
            results
        ))
        
        # Artifacts Detection (New)
        if video.get("artifacts"):
            story.extend(self._build_artifacts_section(video.get("artifacts", {})))
        
        # Flagged Segments
        if segments:
            story.extend(self._build_segments_section(segments))
        
        # Summary
        if summary_text:
            story.extend(self._build_summary_section(summary_text))
        
        # Recommendations
        story.extend(self._build_recommendations_section())
        
        # Limitations
        story.extend(self._build_limitations_section())
        
        # Footer
        story.extend(self._build_footer(job_id))
        
        # Build PDF
        doc.build(story)
        
        return str(filepath)
    
    def _build_header(self, job_id: str) -> List:
        """Build report header."""
        elements = []
        
        # Title
        elements.append(Paragraph(
            "🛡️ DeepFakeShield AI",
            self.styles['ReportTitle']
        ))
        
        elements.append(Paragraph(
            "Forensic Analysis Report",
            self.styles['ReportSubtitle']
        ))
        
        # Report metadata
        meta_data = [
            ["Report ID:", job_id[:8].upper()],
            ["Generated:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["Version:", "2.1.0"]
        ]
        
        meta_table = Table(meta_data, colWidths=[1.5*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), COLORS['muted']),
            ('TEXTCOLOR', (1, 0), (1, -1), COLORS['dark']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(meta_table)
        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['light']))
        
        return elements
    
    def _build_verdict_section(self, score: float, label: str) -> List:
        """Build verdict section with score visualization."""
        elements = []
        
        elements.append(Paragraph("Analysis Verdict", self.styles['DSSectionHeader']))
        
        verdict_text, verdict_desc = get_verdict_text(score, label)
        verdict_color = get_verdict_color(score)
        
        # Verdict box
        verdict_style = ParagraphStyle(
            'VerdictBox',
            parent=self.styles['DSVerdictText'],
            textColor=verdict_color
        )
        
        elements.append(Paragraph(verdict_text, verdict_style))
        elements.append(Paragraph(verdict_desc, self.styles['DSBodyText']))
        elements.append(Spacer(1, 10))
        
        # Score display
        score_pct = int(score * 100)
        score_data = [
            ["Suspicion Score:", f"{score_pct}%"],
            ["Classification:", label or "N/A"]
        ]
        
        score_table = Table(score_data, colWidths=[2*inch, 2*inch])
        score_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (0, -1), COLORS['muted']),
            ('TEXTCOLOR', (1, 0), (1, -1), verdict_color),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _build_technical_section(self, results: Dict) -> List:
        """Build technical summary section."""
        elements = []
        tech = results.get("technical_summary", {})
        meta = results.get("metadata", {})
        
        elements.append(Paragraph("Technical Summary", self.styles['DSSectionHeader']))
        
        data = [
            ["Models Used", ", ".join(tech.get("models_used", []))[:60]],
            ["Inference Time", f"{tech.get('total_inference_time_ms', 0)}ms"],
            ["Resolution", meta.get("resolution", "N/A")],
            ["Codec", meta.get("codec", "N/A")]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), COLORS['muted']),
            ('TEXTCOLOR', (1, 0), (1, -1), COLORS['dark']),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_modality_section(
        self,
        video_score: float,
        audio_score: float,
        lipsync_score: float,
        media_type: str,
        results: Dict
    ) -> List:
        """Build modality analysis section with detailed metrics."""
        elements = []
        
        elements.append(Paragraph("Detailed Forensics", self.styles['DSSectionHeader']))
        
        # Helper to format rows
        def make_row(label, value, is_header=False):
            return [label, value]

        # Video Section
        if media_type in ["video", "image"]:
            vid = results.get("video", {})
            elements.append(Paragraph("🎬 Video Analysis", self.styles['Heading3']))
            v_data = [
                ["Suspicion Score", f"{int(video_score*100)}%"],
                ["Manipulation Type", vid.get("manipulation_type") or "None"],
                ["Faces Detected", str(vid.get("faces_detected", "--"))],
                ["Frames Analyzed", str(vid.get("frames_analyzed", "--"))]
            ]
            t = Table(v_data, colWidths=[2.5*inch, 3*inch])
            t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, COLORS['light'])]))
            elements.append(t)
            elements.append(Spacer(1, 10))

        # Audio Section
        if media_type in ["video", "audio"]:
            aud = results.get("audio", {})
            elements.append(Paragraph("🔊 Audio Analysis", self.styles['Heading3']))
            a_data = [
                ["Suspicion Score", f"{int(audio_score*100)}%"],
                ["Voice Cloning", "Detected" if aud.get("voice_cloning_detected") else "Not Detected"],
                ["Cloning Method", aud.get("cloning_method") or "N/A"],
                ["MFCC Anomaly", f"{float(aud.get('spectral_analysis', {}).get('mfcc_anomaly_score', 0))*100:.1f}%"]
            ]
            t = Table(a_data, colWidths=[2.5*inch, 3*inch])
            t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, COLORS['light'])]))
            elements.append(t)
            elements.append(Spacer(1, 10))

        # Lipsync Section
        if media_type == "video":
            ls = results.get("lipsync", {})
            elements.append(Paragraph("👄 Lip-Sync Analysis", self.styles['Heading3']))
            l_data = [
                ["Suspicion Score", f"{int(lipsync_score*100)}%"],
                ["Sync Offset", f"{ls.get('sync_offset_ms', 0)}ms"],
                ["Phoneme Accuracy", f"{float(ls.get('phoneme_accuracy', 0))*100:.1f}%"],
                ["Viseme Match", f"{float(ls.get('viseme_match_rate', 0))*100:.1f}%"]
            ]
            t = Table(l_data, colWidths=[2.5*inch, 3*inch])
            t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, COLORS['light'])]))
            elements.append(t)
            elements.append(Spacer(1, 10))
            
        # --- NEW FORENSIC SECTIONS ---
        if 'media_quality' in results:
             elements.extend(self._build_media_quality_section(results['media_quality']))
             
        if 'frequency_analysis' in results:
             elements.extend(self._build_frequency_analysis_section(results['frequency_analysis']))

        if 'container_analysis' in results:
             elements.extend(self._build_container_forensics_section(results['container_analysis']))

        if 'linguistic_analysis' in results:
             elements.extend(self._build_linguistic_analysis_section(results['linguistic_analysis']))

        return elements

    def _build_linguistic_analysis_section(self, ling: Dict) -> List:
        elements = []
        elements.append(Paragraph("Linguistic & Speech Pattern Analysis", self.styles['DSSectionHeader']))
        
        fluency = ling.get("fluency_score", 0)
        prob = ling.get("generated_text_probability", 0)
        patterns = ling.get("suspicious_patterns", {})
        
        data = [
            ["AI Text Probability", f"{int(prob*100)}%"],
            ["Templated Speech", "⚠️ DETECTED" if patterns.get("templated_speech") else "None"],
            ["Unnatural Repetition", "⚠️ DETECTED" if patterns.get("unnatural_repetition") else "None"],
            ["Sentiment Consistency", "Inconsistent" if patterns.get("sentiment_inconsistency") else "Consistent"]
        ]
        
        t = Table(data, colWidths=[2.5*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), COLORS['light']),
            ('GRID', (0,0), (-1,-1), 0.5, COLORS['light']),
            ('TEXTCOLOR', (1,1), (1,1), COLORS['error'] if "DETECTED" in data[1][1] or "DETECTED" in data[2][1] else COLORS['dark'])
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))
        return elements

    def _build_media_quality_section(self, quality: Dict) -> List:
        elements = []
        elements.append(Paragraph("Media Quality & Integrity", self.styles['DSSectionHeader']))
        
        score = quality.get("overall_quality_score", 0)
        data = [
            ["Quality Score", f"{score}/100"],
            ["Blur Detected", "Yes" if quality.get("blur_detection", {}).get("is_blurry") else "No"],
            ["Noise Level", f"{quality.get('noise_level', {}).get('snr_db', 0)} dB SNR"],
            ["Double Compression", "Detected" if quality.get("compression_analysis", {}).get("double_compression_detected") else "None"]
        ]
        
        t = Table(data, colWidths=[2.5*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), COLORS['light']),
            ('GRID', (0,0), (-1,-1), 0.5, COLORS['light']),
            ('TEXTCOLOR', (1,1), (1,1), COLORS['error'] if "Yes" in data[1][1] else COLORS['success'])
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))
        return elements

    def _build_frequency_analysis_section(self, freq: Dict) -> List:
        elements = []
        elements.append(Paragraph("Frequency Domain Forensics", self.styles['DSSectionHeader']))
        
        gan_detected = freq.get("gan_fingerprint_detected", False)
        data = [
            ["GAN Fingerprint", "⚠️ DETECTED" if gan_detected else "✓ None"],
            ["Spectrum Consistency", freq.get("spectrum_consistency", "NORMAL")],
            ["High Freq Artifacts", "Present" if freq.get("fft_anomalies", {}).get("high_freq_artifacts") else "Absent"]
        ]
        
        t = Table(data, colWidths=[2.5*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), COLORS['light']),
            ('GRID', (0,0), (-1,-1), 0.5, COLORS['light']),
            ('TEXTCOLOR', (1,0), (1,0), COLORS['error'] if gan_detected else COLORS['success'])
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))
        return elements

    def _build_container_forensics_section(self, container: Dict) -> List:
        elements = []
        elements.append(Paragraph("Container & Metadata Forensics", self.styles['DSSectionHeader']))
        
        tools = ", ".join(container.get("tool_fingerprints", []))
        data = [
            ["Metadata Consistency", container.get("metadata_consistency", "UNKNOWN")],
            ["Tool Fingerprints", tools],
            ["Date Modification Mismatch", "Yes" if container.get("modification_date_mismatch") else "No"]
        ]
        
        t = Table(data, colWidths=[2.5*inch, 4*inch]) # Wider for tools
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), COLORS['light']),
            ('GRID', (0,0), (-1,-1), 0.5, COLORS['light']),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))
        return elements

    def _build_artifacts_section(self, artifacts: Dict) -> List:
        """Build artifacts detection section."""
        elements = []
        elements.append(Paragraph("Artifact Detection", self.styles['DSSectionHeader']))
        
        data = [["Artifact Type", "Status"]]
        
        style_cmds = [
            ('BACKGROUND', (0,0), (-1,0), COLORS['muted']),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, COLORS['light']),
        ]
        
        row_idx = 1
        for k, v in artifacts.items():
            name = k.replace("_", " ").title()
            
            # If simulation returns True for artifacts, that means artifact IS present (BAD).
            status_text = "⚠️ DETECTED" if v else "✓ Cleared"
            text_color = COLORS['error'] if v else COLORS['success']
            
            data.append([name, status_text])
            
            # Add dynamic color style for this row
            style_cmds.append(('TEXTCOLOR', (1, row_idx), (1, row_idx), text_color))
            row_idx += 1

        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle(style_cmds))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        return elements

    def _build_segments_section(self, segments: List[Dict]) -> List:
        """Build flagged segments timeline section."""
        elements = []
        elements.append(Paragraph("Suspicious Segments Timeline", self.styles['DSSectionHeader']))
        
        data = [["Time Range", "Type", "Score"]]
        
        style_cmds = [
             ('BACKGROUND', (0,0), (-1,0), COLORS['muted']),
             ('TEXTCOLOR', (0,0), (-1,0), colors.white),
             ('GRID', (0,0), (-1,-1), 0.5, COLORS['light']),
        ]
        
        row_idx = 1
        for seg in segments:
            start = seg.get('start_ms', 0) / 1000.0
            end = seg.get('end_ms', 0) / 1000.0
            score = seg.get('score', 0)
            stype = seg.get('segment_type', 'unknown').upper()
            
            data.append([
                f"{start:.1f}s - {end:.1f}s",
                stype,
                f"{int(score*100)}%"
            ])
            
            if score > 0.7:
                 style_cmds.append(('TEXTCOLOR', (2, row_idx), (2, row_idx), COLORS['error']))
            elif score > 0.4:
                 style_cmds.append(('TEXTCOLOR', (2, row_idx), (2, row_idx), COLORS['warning']))
            
            row_idx += 1
            
        t = Table(data, colWidths=[2*inch, 2*inch, 1.5*inch])
        t.setStyle(TableStyle(style_cmds))
        elements.append(t)
        elements.append(Spacer(1, 20))
        return elements

    def _build_summary_section(self, text: str) -> List:
        """Build summary section."""
        elements = []
        elements.append(Paragraph("Forensic Summary", self.styles['DSSectionHeader']))
        elements.append(Paragraph(text, self.styles['DSBodyText']))
        elements.append(Spacer(1, 20))
        return elements

    def _build_recommendations_section(self) -> List:
        """Build recommendations."""
        elements = []
        elements.append(Paragraph("Recommendations", self.styles['DSSectionHeader']))
        
        recs = [
            "• Verify the original source of the media file.",
            "• Check for metadata consistency using external tools.",
            "• Use this report as a preliminary screening tool, not absolute proof."
        ]
        
        for r in recs:
            elements.append(Paragraph(r, self.styles['DSBodyText']))
        
        elements.append(Spacer(1, 15))
        return elements

    def _build_limitations_section(self) -> List:
        """Build limitations."""
        elements = []
        elements.append(Paragraph("Limitations & Disclaimer", self.styles['DSSectionHeader']))
        
        lims = [
            "• Deepfake detection models are probabilistic and may produce false positives/negatives.",
            "• Performance may degrade on highly compressed or low-resolution media.",
            "• This report does not constitute legal evidence."
        ]
        
        for l in lims:
            elements.append(Paragraph(l, self.styles['DSBodyText']))
            
        elements.append(Spacer(1, 20))
        return elements

    def _build_footer(self, job_id: str) -> List:
        """Build footer."""
        elements = []
        elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['light']))
        elements.append(Spacer(1, 10))
        
        text = f"DeepFakeShield Analysis Report • ID: {job_id} • {datetime.utcnow().year}"
        elements.append(Paragraph(text, self.styles['ReportSubtitle']))
        return elements

# Singleton instance
pdf_generator = PDFReportGenerator()


def generate_pdf_report(
    job_id: str,
    overall_score: float,
    label: str,
    results: Dict[str, Any] = None,   # Changed signature to accept full results
    summary_text: str = None,
    media_type: str = "video",
    # Legacy args for backward compatibility if needed, but better to use results dict
    video_score: float = 0.0,
    audio_score: float = 0.0,
    lipsync_score: float = 0.0,
    segments: List[Dict] = None
) -> str:
    """
    Generate a PDF report for the given analysis results.
    """
    return pdf_generator.generate_report(
        job_id=job_id,
        overall_score=overall_score,
        label=label,
        results=results,
        summary_text=summary_text,
        media_type=media_type
    )
