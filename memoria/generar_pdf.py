"""Compone la memoria desde Markdown con índice y paginado verificable."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'), help='Directorio con arial.ttf, arialbd.ttf, ariali.ttf y arialbi.ttf.')
FONT_ROOT = parser.parse_args().font_dir
for name, filename in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'), ('Arial-Italic', 'ariali.ttf'), ('Arial-BoldItalic', 'arialbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / filename)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-BoldItalic')

INK = colors.HexColor('#243946')
TEXT = colors.HexColor('#20252A')
styles = {
    'body': ParagraphStyle('Body', fontName='Arial', fontSize=11, leading=14.3, textColor=TEXT, alignment=TA_JUSTIFY, spaceAfter=7, allowWidows=0, allowOrphans=0),
    'h2': ParagraphStyle('H2', fontName='Arial-Bold', fontSize=14, leading=18, textColor=INK, spaceBefore=16, spaceAfter=10, keepWithNext=True),
    'h3': ParagraphStyle('H3', fontName='Arial-Bold', fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=7, keepWithNext=True),
    'cell': ParagraphStyle('Cell', fontName='Arial', fontSize=9.5, leading=12.2, textColor=TEXT, spaceAfter=0),
    'caption': ParagraphStyle('Caption', fontName='Arial-Italic', fontSize=9.5, leading=12.2, textColor=INK, spaceAfter=10),
    'bib': ParagraphStyle('Bibliography', fontName='Arial', fontSize=10, leading=12, textColor=TEXT, spaceAfter=4),
    'title': ParagraphStyle('Title', fontName='Arial-Bold', fontSize=26, leading=32, textColor=INK, spaceAfter=20),
    'subtitle': ParagraphStyle('Subtitle', fontName='Arial', fontSize=17, leading=23, textColor=INK, spaceAfter=36),
    'meta': ParagraphStyle('Metadata', fontName='Arial', fontSize=11, leading=17, textColor=TEXT, spaceAfter=8),
}


def inline(text):
    text = html.escape(text)
    # Componer exponentes con glifos básicos evita caracteres ausentes en Arial.
    superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+', lambda m: '<super>' + m.group().translate(superscripts) + '</super>', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#245979">\1</link>', text)
    text = re.sub(r'`([^`]+)`', r'<font size="9.4">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    return text


class Thesis(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=24*mm, rightMargin=24*mm, topMargin=23*mm, bottomMargin=22*mm,
                         title='Evaluación empírica de métricas de AFT8-15 en Mistral', author='Francisco Jose Martinez Fernandez')
        self.addPageTemplates(PageTemplate('main', [Frame(self.leftMargin, self.bottomMargin, self.width, self.height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.draw_page))
        self.heading_pages = {}

    def draw_page(self, canvas, doc):
        if doc.page <= 2:
            return
        canvas.saveState()
        canvas.setFont('Arial', 8)
        canvas.setFillColor(colors.HexColor('#66727C'))
        canvas.drawString(self.leftMargin, A4[1]-15*mm, 'TFM · AFT8-15 · Evaluación empírica de AFT8-15')
        canvas.drawRightString(A4[0]-self.rightMargin, 13*mm, str(doc.page-2))
        canvas.restoreState()

    def afterFlowable(self, item):
        if isinstance(item, Paragraph) and getattr(item, 'toc_key', None):
            title = item.getPlainText()
            self.canv.bookmarkPage(item.toc_key)
            self.canv.addOutlineEntry(title, item.toc_key, level=0)
            self.notify('TOCEntry', (0, title, self.page-2, item.toc_key))
            self.heading_pages[title] = self.page-2


def main():
    source = ROOT / 'TFM_memoria_revisada.md'
    text = source.read_text(encoding='utf-8')
    front, body = text.split('## Resumen ejecutivo', 1)
    story = [Spacer(1, 34*mm)]
    for block in front.strip().split('\n\n'):
        if block.startswith('# '):
            story.append(Paragraph(inline(block[2:]), styles['title']))
        elif block.startswith('## '):
            story.append(Paragraph(inline(block[3:]), styles['subtitle']))
        else:
            story.append(Paragraph(inline(block), styles['meta']))
    story += [PageBreak(), Paragraph('Índice de contenidos', styles['h2']), Spacer(1, 8*mm)]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('TOC', fontName='Arial', fontSize=11, leading=17, spaceBefore=9, leftIndent=0, rightIndent=12)]
    story += [toc, PageBreak()]
    lines = ('## Resumen ejecutivo' + body).splitlines()
    bibliography = False
    bibliography_start = None
    i, heading_no = 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('## '):
            heading_no += 1
            label = line[3:]
            if label == '1. Contexto, necesidad y aportación individual':
                story.append(PageBreak())
            bibliography = label == 'Bibliografía'
            if bibliography:
                bibliography_start = len(story)
            heading = Paragraph(inline(label), styles['h2'])
            heading.toc_key = f'section-{heading_no}'
            story.append(heading)
            i += 1
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), styles['h3']))
            i += 1
        elif line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip()); i += 1
            rows = [[part.strip() for part in row.strip('|').split('|')] for row in table_lines]
            rows = [row for row in rows if not all(re.fullmatch(r'[:\- ]+', cell) for cell in row)]
            n = len(rows[0]); width = A4[0]-48*mm
            if n == 2:
                widths = [width*.34, width*.66]
            elif n == 3:
                widths = [width*.43, width*.285, width*.285]
            elif n == 4:
                widths = [width*.40, width*.20, width*.20, width*.20]
            else:
                widths = [width*.28] + [width*.72/(n-1)]*(n-1)
            cells = [[Paragraph(inline(value) if r else '<b>'+inline(value)+'</b>', styles['cell']) for value in row] for r,row in enumerate(rows)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF2')), ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7), ('LINEBELOW',(0,0),(-1,0),.7,INK), ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D8DEE3'))]))
            story += [KeepTogether([table]), Spacer(1, 10)]
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*)\]\((.*)\)', line)
            caption, file = match.groups()
            path = ROOT / file
            with PILImage.open(path) as im:
                w,h = im.size
            scale = min((A4[0]-48*mm)/w, 125*mm/h)
            story.append(KeepTogether([Image(str(path), width=w*scale, height=h*scale), Spacer(1,5), Paragraph(inline(caption), styles['caption'])]))
            i += 1
        else:
            parts = [line]; i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
                parts.append(lines[i].strip()); i += 1
            story.append(Paragraph(inline(' '.join(parts)), styles['bib'] if bibliography else styles['body']))
    if bibliography_start is not None:
        story[bibliography_start:] = [KeepTogether(story[bibliography_start:])]
    output = ROOT / 'Francisco_Jose_Martinez_Fernandez_TFM_AFT8_15.pdf'
    document = Thesis(output)
    document.multiBuild(story)
    pages = PdfReader(output).pages
    body_pages = len(pages)-2
    report = {'total_pages': len(pages), 'cover_pages': 1, 'contents_pages': 1, 'body_including_bibliography': body_pages, 'limit':20, 'within_limit': body_pages <= 20, 'font':'Arial', 'body_font_pt':11, 'leading_pt':14.3, 'margins_mm':{'left':24,'right':24,'top':23,'bottom':22}, 'section_pages':document.heading_pages, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'pdf_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT/'paginacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if body_pages > 20:
        raise SystemExit('La memoria supera el límite de 20 caras.')


if __name__ == '__main__':
    main()
