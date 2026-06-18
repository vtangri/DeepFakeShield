import sys
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas

# ──────────────────────────────────────────────────────────────────────
# Dual-Pass Canvas to calculate total pages & render professional headers/footers
# ──────────────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Exclude cover page from headers/footers
        if self._pageNumber == 1:
            return
            
        self.saveState()
        
        # Colors
        c_muted = colors.HexColor("#64748B")
        c_line = colors.HexColor("#E2E8F0")
        
        # 1. Running Header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#6366F1")) # Indigo
        self.drawString(54, 790, "🛡️ DEEPFAKSHIELD AI")
        self.setFont("Helvetica", 8)
        self.setFillColor(c_muted)
        self.drawRightString(541, 790, "Forensic Platform & Dissertation Documentation")
        
        # Thin header divider rule
        self.setStrokeColor(c_line)
        self.setLineWidth(0.5)
        self.line(54, 782, 541, 782)
        
        # 2. Running Footer
        self.line(54, 52, 541, 52)
        self.drawString(54, 40, "Confidential • Academic Forensic Integrity")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 40, page_text)
        
        self.restoreState()

# ──────────────────────────────────────────────────────────────────────
# Markdown to PDF Compiler Engine
# ──────────────────────────────────────────────────────────────────────
def compile_md_to_pdf(md_path, pdf_path):
    # Setup document template with 1-inch top/bottom margin for header/footer clearance
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Elegant Color Palette
    c_primary = colors.HexColor('#0F172A')   # Slate 900
    c_accent = colors.HexColor('#6366F1')    # Indigo 500
    c_text = colors.HexColor('#334155')      # Slate 700
    c_muted = colors.HexColor('#64748B')     # Slate 500
    c_bg = colors.HexColor('#F8FAFC')        # Slate 50
    c_border = colors.HexColor('#E2E8F0')    # Slate 200
    
    # Styling Configuration
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=32,
        spaceAfter=10,
        textColor=c_primary,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        spaceAfter=25,
        textColor=c_accent,
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        spaceBefore=18,
        spaceAfter=8,
        textColor=c_accent,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        textColor=c_primary,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.0,
        spaceAfter=8,
        textColor=c_text,
        alignment=TA_JUSTIFY
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
        textColor=c_text
    )
    
    quote_style = ParagraphStyle(
        'DocQuote',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14.0,
        textColor=c_text
    )
    
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor('#F8FAFC')
    )
    
    equation_style = ParagraphStyle(
        'DocEquation',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=12
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=c_accent
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=c_primary
    )
    
    story = []
    
    # Read file
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_content = []
    in_table = False
    table_data = []
    
    def process_inline_md(text):
        t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        t = re.sub(r'\*(.*?)\*', r'<i>\1</i>', t)
        t = re.sub(r'`(.*?)`', r'<font name="Courier" color="#6366F1">\1</font>', t)
        return t

    i = 0
    is_first_header = True
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # --- Code Blocks ---
        if stripped.startswith('```'):
            if in_code_block:
                # Compile code snippet in premium dark panel
                code_text = '\n'.join(code_content)
                t = Table([[Preformatted(code_text, code_style)]], colWidths=[487])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0F172A')), # Dark slate background
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
                in_code_block = False
                code_content = []
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_content.append(line.rstrip('\n'))
            i += 1
            continue
            
        # --- Tables ---
        if stripped.startswith('|') and stripped.endswith('|'):
            if not in_table:
                in_table = True
                table_data = []
            
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if not all(re.match(r'^[-:]+$', c) for c in cells):
                table_data.append(cells)
            i += 1
            continue
        elif in_table:
            # Compile Table Flowable
            if table_data:
                formatted_table_data = []
                for row_idx, row in enumerate(table_data):
                    formatted_row = []
                    for col_idx, cell in enumerate(row):
                        cell_text_style = ParagraphStyle(
                            f'Cell_{row_idx}_{col_idx}',
                            parent=body_style,
                            fontName='Helvetica-Bold' if row_idx == 0 else 'Helvetica',
                            fontSize=8.5,
                            leading=11,
                            textColor=colors.white if row_idx == 0 else c_primary
                        )
                        formatted_row.append(Paragraph(process_inline_md(cell), cell_text_style))
                    formatted_table_data.append(formatted_row)
                
                num_cols = len(table_data[0])
                if num_cols == 2:
                    col_widths = [140, 347]
                elif num_cols == 3:
                    col_widths = [110, 110, 267]
                else:
                    col_widths = [487 / num_cols] * num_cols
                    
                t = Table(formatted_table_data, colWidths=col_widths, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), c_accent),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, c_border),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
            in_table = False
            
        if stripped == '':
            i += 1
            continue
            
        # --- Math Equations ---
        if stripped.startswith('$$') and stripped.endswith('$$'):
            equation = stripped[2:-2].strip()
            # Clean fraction and summation notations for readable text display
            equation = equation.replace(r'\frac', '').replace(r'\sum', 'Σ').replace('{', '(').replace('}', ')').replace(r'\times', ' × ')
            story.append(Paragraph(equation, equation_style))
            i += 1
            continue
            
        # --- Headers ---
        if stripped.startswith('# '):
            # Create a Cover Page
            story.append(Spacer(1, 120))
            story.append(Paragraph(process_inline_md(stripped[2:]), title_style))
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="60%", thickness=2, color=c_accent, spaceBefore=4, spaceAfter=20, hAlign='CENTER'))
            is_first_header = False
            i += 1
            continue
        elif stripped.startswith('## '):
            if is_first_header or len(story) <= 5:
                # Subtitle on Cover Page
                story.append(Paragraph(process_inline_md(stripped[3:]), subtitle_style))
                
                # Build Elegant Metadata Card on Cover Page
                meta_rows = []
                meta_idx = i + 1
                meta_collected = {}
                while meta_idx < len(lines) and len(meta_collected) < 3:
                    m_line = lines[meta_idx].strip()
                    if m_line.startswith('> '):
                        m_text = m_line[2:]
                        if 'Author:' in m_text:
                            meta_collected['Author'] = m_text.replace('**Author:**', '').strip()
                        elif 'Course:' in m_text:
                            meta_collected['Course'] = m_text.replace('**Course:**', '').strip()
                        elif 'Objective:' in m_text:
                            meta_collected['Objective'] = m_text.replace('**Objective:**', '').strip()
                    meta_idx += 1
                
                if meta_collected:
                    m_data = []
                    for label, val in meta_collected.items():
                        m_data.append([Paragraph(label, meta_label_style), Paragraph(val, meta_val_style)])
                    
                    meta_table = Table(m_data, colWidths=[90, 270])
                    meta_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
                        ('GRID', (0,0), (-1,-1), 0.5, c_border),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('LEFTPADDING', (0, 0), (-1, -1), 12),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ]))
                    
                    outer_table = Table([[meta_table]], colWidths=[380], hAlign='CENTER')
                    outer_table.setStyle(TableStyle([
                        ('LINELEFT', (0,0), (0,-1), 4, c_accent),
                        ('BACKGROUND', (0,0), (-1,-1), c_bg),
                        ('TOPPADDING', (0,0), (-1,-1), 1),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
                        ('LEFTPADDING', (0,0), (-1,-1), 1),
                        ('RIGHTPADDING', (0,0), (-1,-1), 1),
                    ]))
                    
                    story.append(Spacer(1, 40))
                    story.append(outer_table)
                
                story.append(PageBreak())
                i = meta_idx
                continue
            else:
                story.append(Paragraph(process_inline_md(stripped[3:]), h1_style))
            i += 1
            continue
        elif stripped.startswith('### '):
            story.append(Paragraph(process_inline_md(stripped[4:]), h2_style))
            i += 1
            continue
            
        # --- Horizontal Rules ---
        if stripped == '---':
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=12))
            i += 1
            continue
            
        # --- Blockquotes ---
        if stripped.startswith('> '):
            quote_text = stripped[2:]
            while i + 1 < len(lines) and lines[i+1].strip().startswith('> '):
                i += 1
                quote_text += ' ' + lines[i].strip()[2:]
            
            q_para = Paragraph(process_inline_md(quote_text), quote_style)
            q_table = Table([[q_para]], colWidths=[487])
            q_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), c_bg),
                ('LINELEFT', (0, 0), (0, -1), 3, c_accent),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(q_table)
            story.append(Spacer(1, 6))
            i += 1
            continue
            
        # --- Bullet Points ---
        if stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped):
            prefix = '• ' if (stripped.startswith('- ') or stripped.startswith('* ')) else ''
            content = stripped[2:] if (stripped.startswith('- ') or stripped.startswith('* ')) else stripped[stripped.find('.')+2:]
            story.append(Paragraph(f"{prefix}{process_inline_md(content)}", bullet_style))
            i += 1
            continue
            
        # --- Standard Paragraph ---
        story.append(Paragraph(process_inline_md(stripped), body_style))
        i += 1
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Premium high-quality PDF compiled successfully at: {pdf_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compile_pdf.py <input_md> <output_pdf>")
        sys.exit(1)
    compile_md_to_pdf(sys.argv[1], sys.argv[2])
