"""Gera o relatório de resultados em A4, formatado conforme a ABNT (NBR 14724), em .docx e .doc.

Fontes (nada é digitado à mão no documento):
  - report/resultados.md                  texto, seções e tabelas do relatório
  - results/experiments*.csv              tabelas completas de experimentos (apêndice)
  - results/figures/*.png                 figuras (apêndice)

Formatação: A4; margens 3 cm (sup./esq.) e 2 cm (inf./dir.); Times New Roman 12; entrelinhas
1,5; texto justificado com recuo de 1,25 cm; numeração de páginas no canto superior direito a
partir da primeira página textual; legendas, fontes e tabelas em corpo 10 e espaçamento simples.

Uso: .venv/bin/python -m src.generate_report [--out report/relatorio_resultados_abnt]
"""
import argparse
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from . import config

FONT = "Times New Roman"
MONO = "Courier New"

# Metadados da capa/folha de rosto; ajuste aqui se necessário.
INSTITUICAO = "INSTITUTO FEDERAL DE GOIÁS"
DISCIPLINA = "Deep Learning"
LOCAL = "Goiás"
ANO = date.today().year
NATUREZA = (
    "Trabalho final da disciplina de Deep Learning, apresentado ao Instituto Federal de "
    "Goiás como requisito de avaliação"
)

REFERENCIAS = [
    "AGÊNCIA NACIONAL DO PETRÓLEO, GÁS NATURAL E BIOCOMBUSTÍVEIS (Brasil). <b>Série histórica "
    "do levantamento de preços de combustíveis</b>. Brasília, DF: ANP. Disponível em: "
    "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/"
    "precos-revenda-e-de-distribuicao-combustiveis/serie-historica-do-levantamento-de-precos. "
    "Acesso em: {acesso}.",
    "AKIBA, T. <i>et al.</i> Optuna: a next-generation hyperparameter optimization framework. "
    "In: ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 25., 2019, "
    "Anchorage. <b>Proceedings</b> [...]. New York: ACM, 2019. p. 2623-2631.",
    "BENGIO, Y.; SIMARD, P.; FRASCONI, P. Learning long-term dependencies with gradient descent "
    "is difficult. <b>IEEE Transactions on Neural Networks</b>, v. 5, n. 2, p. 157-166, 1994.",
    "CANUTO, S. <b>Redes neurais recorrentes e tratamento de sequências</b>. Goiânia: Instituto "
    "Federal de Goiás, 2026. Material de aula (slides) da disciplina de Deep Learning.",
    "CHO, K. <i>et al.</i> Learning phrase representations using RNN encoder-decoder for "
    "statistical machine translation. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE "
    "PROCESSING, 2014, Doha. <b>Proceedings</b> [...]. Stroudsburg: ACL, 2014. p. 1724-1734.",
    "CHOLLET, F. <b>Deep learning with Python</b>. Shelter Island: Manning, 2018.",
    "CHUNG, J. <i>et al.</i> Empirical evaluation of gated recurrent neural networks on sequence "
    "modeling. <b>arXiv</b>, 1412.3555, 2014. Disponível em: https://arxiv.org/abs/1412.3555. "
    "Acesso em: {acesso}.",
    "ELMAN, J. L. Finding structure in time. <b>Cognitive Science</b>, v. 14, n. 2, p. 179-211, 1990.",
    "HOCHREITER, S.; SCHMIDHUBER, J. Long short-term memory. <b>Neural Computation</b>, "
    "Cambridge, v. 9, n. 8, p. 1735-1780, 1997.",
    "HYNDMAN, R. J.; ATHANASOPOULOS, G. <b>Forecasting</b>: principles and practice. 3. ed. "
    "Melbourne: OTexts, 2021. Disponível em: https://otexts.com/fpp3/. Acesso em: {acesso}.",
    "MEESE, R. A.; ROGOFF, K. Empirical exchange rate models of the seventies: do they fit out of "
    "sample? <b>Journal of International Economics</b>, v. 14, n. 1-2, p. 3-24, 1983.",
    "PASCANU, R.; MIKOLOV, T.; BENGIO, Y. On the difficulty of training recurrent neural networks. "
    "In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 30., 2013, Atlanta. <b>Proceedings</b> [...]. "
    "[S. l.]: PMLR, 2013. p. 1310-1318.",
    "PEDREGOSA, F. <i>et al.</i> Scikit-learn: machine learning in Python. <b>Journal of Machine "
    "Learning Research</b>, v. 12, p. 2825-2830, 2011.",
]

MESES = ["jan.", "fev.", "mar.", "abr.", "maio", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]

TABLE_CAPTIONS = {
    "Split|MAE (R$/L)|RMSE (R$/L)|R²": "Desempenho do baseline de persistência (R$/L)",
    "Janela|SimpleRNN|GRU|LSTM": "MAE de validação (R$/L) por janela e arquitetura na comparação principal",
    "#|Config.|Unid.|Camadas": "Três melhores configurações por MAE de validação",
    "Split|Preço mínimo (R$/L)|Preço máximo (R$/L)|Média (R$/L)": "Faixa de preços da gasolina em cada partição (R$/L)",
    "Modelo|MAE val|MAE teste|RMSE teste": "Comparação entre as formulações em nível e em log-retorno",
}

MODEL_NAMES = {"SimpleRNN": "SimpleRNN", "GRU": "GRU", "LSTM": "LSTM"}
EXPERIMENT_COLUMNS = {
    "tag": "Grupo", "modelo": "Modelo", "janela": "Jan.", "unidades": "Unid.", "camadas": "Cam.",
    "dropout": "Dropout", "l2": "L2", "parametros": "Parâm.", "epocas": "Épocas",
    "mae_val": "MAE val", "rmse_val": "RMSE val", "r2_val": "R² val", "mae_test": "MAE teste",
    "rmse_test": "RMSE teste", "r2_test": "R² teste", "tempo_treino_s": "Tempo (s)",
}
TAG_LABELS = {
    "principal": "comparação principal",
    "unidades": "grade de unidades",
    "profundidade": "grade de profundidade",
    "regularizacao": "grade de regularização",
    "optuna_best": "melhor busca Optuna",
}
FIG_KINDS = {
    "curvas": "Curvas de treinamento e validação",
    "real_vs_previsto": "Valor real × previsto no conjunto de teste",
    "residuos": "Resíduos ao longo do tempo e sua distribuição",
}


# --------------------------------------------------------------------------------------
# Utilidades de baixo nível (XML do Word)
# --------------------------------------------------------------------------------------
def _set_run_font(run, name=FONT, size=None, bold=None, italic=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def _style_font(style, size, bold=False, italic=False, caps=False):
    font = style.font
    font.name = FONT
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    font.all_caps = caps
    font.color.rgb = None
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in list(rfonts.attrib):
        del rfonts.attrib[attr]
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)
    color = rpr.find(qn("w:color"))
    if color is not None:
        rpr.remove(color)


def _add_field(paragraph, instruction, placeholder="", size=None):
    """Insere um campo do Word (PAGE, TOC...) com texto provisório."""
    def _run(child):
        r = paragraph.add_run()
        _set_run_font(r, size=size)
        r._element.append(child)
        return r

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    _run(begin)
    _run(instr)
    _run(sep)
    r = paragraph.add_run(placeholder)
    _set_run_font(r, size=size)
    _run(end)


def _set_cell_borders(cell, **edges):
    tc_pr = cell._element.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, sz in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def _set_cell_width(cell, width_cm):
    cell.width = Cm(width_cm)


def _repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    tr_pr.append(el)
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)


# --------------------------------------------------------------------------------------
# Parser de Markdown (subconjunto usado em report/resultados.md)
# --------------------------------------------------------------------------------------
INLINE_RE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<math>(?<![\w$])\$(?=\S).+?(?<=\S)\$(?![\w$]))"
    r"|(?P<bold>\*\*.+?\*\*)"
    r"|(?P<link>\[[^\]]+\]\([^)]+\))"
    r"|(?P<ital>(?<!\w)_(?=\S).+?(?<=\S)_(?!\w))"
)


def parse_markdown(text: str):
    lines = text.splitlines()
    blocks, front, i = [], {}, 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
        elif stripped.startswith("# "):
            blocks.append(("title", stripped[2:].strip()))
            i += 1
        elif stripped.startswith(">"):
            m = re.match(r">\s*\*\*(.+?)\*\*:?\s*(.*)", stripped)
            if m:
                front[m.group(1).rstrip(":").strip()] = m.group(2).strip()
            i += 1
        elif stripped.startswith("### "):
            blocks.append(("h2", stripped[4:].strip()))
            i += 1
        elif stripped.startswith("## "):
            blocks.append(("h1", stripped[3:].strip()))
            i += 1
        elif stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            aligns = ["right" if c.endswith(":") else "left" for c in rows[1]]
            blocks.append(("table", rows[0], rows[2:], aligns))
        elif stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip():
                cur = lines[i]
                if cur.startswith("- "):
                    items.append(cur[2:].strip())
                elif cur.startswith("  ") and items:
                    items[-1] += " " + cur.strip()
                else:
                    break
                i += 1
            blocks.append(("bullets", items))
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|>|\||- )", lines[i]):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(para)))
    return blocks, front


# --------------------------------------------------------------------------------------
# Renderização inline
# --------------------------------------------------------------------------------------
def _add_text(par, text, size, bold=False, italic=False, font=FONT, subscript=False):
    if not text:
        return
    run = par.add_run(text)
    _set_run_font(run, name=font, size=size, bold=bold or None, italic=italic or None)
    if subscript:
        run.font.subscript = True


def _add_math(par, expr, size):
    expr = (
        expr.replace(r"\rightarrow", "→")
        .replace(r"\hat{y}", "ŷ")
        .replace(r"\,", " ")
    )
    pos = 0
    for m in re.finditer(r"_(\{[^}]*\}|.)", expr):
        _add_text(par, expr[pos:m.start()], size, italic=True)
        sub = m.group(1)
        _add_text(par, sub[1:-1] if sub.startswith("{") else sub, size, italic=True, subscript=True)
        pos = m.end()
    _add_text(par, expr[pos:], size, italic=True)


def add_inline(par, text, size=12, bold=False, italic=False):
    pos = 0
    for m in INLINE_RE.finditer(text):
        _add_text(par, text[pos:m.start()], size, bold, italic)
        kind, token = m.lastgroup, m.group()
        if kind == "code":
            _add_text(par, token[1:-1], size - 2, bold, False, font=MONO)
        elif kind == "math":
            _add_math(par, token[1:-1], size)
        elif kind == "bold":
            add_inline(par, token[2:-2], size, True, italic)
        elif kind == "ital":
            add_inline(par, token[1:-1], size, bold, True)
        elif kind == "link":
            label, url = re.match(r"\[(.+?)\]\((.+?)\)", token).groups()
            plain = re.sub(r"^_|_$", "", label)
            add_inline(par, label, size, bold, italic)
            if url.startswith("http") and plain.strip("`") != url:
                _add_text(par, f" ({url})", size, bold, italic)
        pos = m.end()
    _add_text(par, text[pos:], size, bold, italic)


def plain_text(text):
    """Texto sem marcação (para títulos e chaves)."""
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)
    return re.sub(r"[`*]|(?<!\w)_|_(?!\w)", "", text).strip()


# --------------------------------------------------------------------------------------
# Construção do documento
# --------------------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.doc = Document()
        self.tab_n = 0
        self.fig_n = 0
        self.h1_n = 0
        self.h2_n = 0
        self.acesso = self._data_acesso()
        self._setup_styles()
        self._setup_section(self.doc.sections[0])

    @staticmethod
    def _data_acesso():
        d = date.today()
        return f"{d.day} {MESES[d.month - 1]} {d.year}"

    # -- estilos e página --------------------------------------------------------------
    def _setup_styles(self):
        st = self.doc.styles
        normal = st["Normal"]
        _style_font(normal, 12)
        pf = normal.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

        for name, caps, bold in (("Heading 1", True, True), ("Heading 2", False, True)):
            s = st[name]
            _style_font(s, 12, bold=bold, caps=caps)
            s.paragraph_format.space_before = Pt(18)
            s.paragraph_format.space_after = Pt(18)
            s.paragraph_format.keep_with_next = True
            s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

        settings = self.doc.settings.element
        upd = OxmlElement("w:updateFields")
        upd.set(qn("w:val"), "true")
        settings.append(upd)

    @staticmethod
    def _setup_section(sec, landscape=False):
        if landscape:
            sec.orientation = WD_ORIENT.LANDSCAPE
            sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
        else:
            sec.orientation = WD_ORIENT.PORTRAIT
            sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.top_margin, sec.left_margin = Cm(3), Cm(3)
        sec.bottom_margin, sec.right_margin = Cm(2), Cm(2)
        sec.header_distance = Cm(1.25)
        sec.footer_distance = Cm(1.25)

    @property
    def text_width_cm(self):
        sec = self.doc.sections[-1]
        return (sec.page_width - sec.left_margin - sec.right_margin) / 360000

    # -- parágrafos --------------------------------------------------------------------
    def para(self, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, size=12, bold=False,
             spacing=1.5, before=0, after=0, keep_next=False, left=None, new_page=False):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.page_break_before = new_page
        pf.alignment = align
        pf.first_line_indent = Cm(1.25) if indent else Cm(0)
        if left is not None:
            pf.left_indent = Cm(left)
        pf.line_spacing = spacing
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        pf.keep_with_next = keep_next
        if text:
            add_inline(p, text, size, bold)
        return p

    def centered(self, text, size=12, bold=False, before=0, after=0, spacing=1.0, new_page=False):
        return self.para(text, WD_ALIGN_PARAGRAPH.CENTER, False, size, bold, spacing, before, after,
                         new_page=new_page)

    def heading(self, level, text):
        if level == 1:
            m = re.match(r"(\d+)\.?\s+(.*)", text)
            self.h1_n = int(m.group(1)) if m else self.h1_n + 1
            self.h2_n = 0
            label = f"{self.h1_n} {(m.group(2) if m else text).upper()}"
        else:
            self.h2_n += 1
            label = f"{self.h1_n}.{self.h2_n} {text}"
        self.doc.add_heading(plain_text(label), level=level)

    def unnumbered_heading(self, text, new_page=True):
        h = self.doc.add_heading(text.upper(), level=1)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        h.paragraph_format.page_break_before = new_page
        return h

    def bullets(self, items):
        for k, item in enumerate(items):
            p = self.para(indent=False, left=1.9)
            p.paragraph_format.first_line_indent = Cm(-0.65)
            add_inline(p, f"{chr(ord('a') + k)}) ", 12)
            add_inline(p, item, 12)

    # -- tabelas -----------------------------------------------------------------------
    def _caption(self, kind, n, title):
        p = self.para(f"{kind} {n} – {title}", WD_ALIGN_PARAGRAPH.CENTER, False, 10,
                      spacing=1.0, before=12, after=6, keep_next=True)
        return p

    def _source(self, text="Elaborado pelos autores"):
        self.para(f"Fonte: {text} ({ANO}).", WD_ALIGN_PARAGRAPH.LEFT, False, 10,
                  spacing=1.0, before=4, after=14)

    def table(self, header, rows, aligns, title, font_size=10, own_section=None):
        wide = len(header) > 8 if own_section is None else own_section
        if wide:
            self._section(landscape=True)
        self.tab_n += 1
        self._caption("Tabela", self.tab_n, title)

        n_cols = len(header)
        table = self.doc.add_table(rows=1 + len(rows), cols=n_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False

        lens = [max(len(plain_text(r[c])) for r in [header] + rows) for c in range(n_cols)]
        weights = [min(max(n, 4), 28) for n in lens]
        total = self.text_width_cm
        widths = [total * w / sum(weights) for w in weights]

        all_rows = [header] + rows
        for r_idx, row in enumerate(all_rows):
            for c_idx, value in enumerate(row):
                cell = table.cell(r_idx, c_idx)
                _set_cell_width(cell, widths[c_idx])
                p = cell.paragraphs[0]
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER if r_idx == 0
                    else WD_ALIGN_PARAGRAPH.RIGHT if aligns[c_idx] == "right"
                    else WD_ALIGN_PARAGRAPH.LEFT
                )
                add_inline(p, value, font_size, bold=(r_idx == 0))
                edges = {}
                if r_idx == 0:
                    edges.update(top=8, bottom=4)
                if r_idx == len(all_rows) - 1:
                    edges.update(bottom=8)
                if edges:
                    _set_cell_borders(cell, **edges)
        _repeat_header(table.rows[0])
        self._source()
        if wide:
            self._section(landscape=False)

    def _section(self, landscape):
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        self._setup_section(sec, landscape)

    # -- figuras -----------------------------------------------------------------------
    def figure(self, path: Path, title):
        self.fig_n += 1
        self._caption("Figura", self.fig_n, title)
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run()
        max_w = min(self.text_width_cm, 15.0)
        pic = run.add_picture(str(path), width=Cm(max_w))
        max_h = Cm(17)
        if pic.height > max_h:
            ratio = max_h / pic.height
            pic.height, pic.width = int(pic.height * ratio), int(pic.width * ratio)
        self._source()

    # -- partes do documento -----------------------------------------------------------
    def cover(self, title, authors):
        self.centered(INSTITUICAO, bold=True, after=6)
        self.centered(f"Disciplina de {DISCIPLINA}", after=0)
        for a in authors:
            self.centered(a.upper(), before=60 if a == authors[0] else 0)
        self.centered(title.upper(), bold=True, before=150)
        self.centered(LOCAL, before=210)
        self.centered(str(ANO))

    def title_page(self, title, authors, professor, link):
        for k, a in enumerate(authors):
            self.centered(a.upper(), new_page=(k == 0))
        self.centered(title.upper(), bold=True, before=140)
        p = self.para(NATUREZA + (f", sob orientação do {professor}." if professor else "."),
                      WD_ALIGN_PARAGRAPH.JUSTIFY, False, 10, spacing=1.0, before=60, left=8)
        if link:
            self.para(f"Repositório: {link}", WD_ALIGN_PARAGRAPH.LEFT, False, 10, spacing=1.0,
                      before=12, left=8)
        self.centered(LOCAL, before=130)
        self.centered(str(ANO))

    def toc(self):
        self.centered("SUMÁRIO", bold=True, after=18, spacing=1.5, new_page=True)
        p = self.doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        _add_field(
            p, r'TOC \o "1-2" \h \z \u',
            "Sumário: atualize o campo (clique com o botão direito > Atualizar campo) para exibir as seções.",
        )

    def start_textual_section(self):
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        self._setup_section(sec)
        header = sec.header
        header.is_linked_to_previous = False
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _add_field(hp, "PAGE", "1", size=10)

    def references(self):
        self.unnumbered_heading("Referências")
        for ref in REFERENCIAS:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(12)
            ref = ref.format(acesso=self.acesso)
            for part in re.split(r"(<b>.*?</b>|<i>.*?</i>)", ref):
                if part.startswith("<b>"):
                    _add_text(p, part[3:-4], 12, bold=True)
                elif part.startswith("<i>"):
                    _add_text(p, part[3:-4], 12, italic=True)
                else:
                    _add_text(p, part, 12)

    def save(self, path: Path):
        self.doc.save(str(path))


# --------------------------------------------------------------------------------------
# Apêndices (dados lidos dos CSVs e das figuras reais)
# --------------------------------------------------------------------------------------
def _fmt(v, nd=4):
    if pd.isna(v):
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}".replace(".", ",")
    return str(v)


def experiments_table(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[[c for c in EXPERIMENT_COLUMNS if c in df.columns]]
    header = [EXPERIMENT_COLUMNS[c] for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in df.columns:
            v = r[c]
            if c in ("janela", "unidades", "camadas", "parametros", "epocas") and not pd.isna(v):
                row.append(f"{int(v):,}".replace(",", ".") if c == "parametros" else str(int(v)))
            elif c == "tempo_treino_s":
                row.append(_fmt(v, 1))
            elif c == "dropout":
                row.append(_fmt(v, 3))
            elif c == "l2":
                row.append("–" if pd.isna(v) else f"{v:g}".replace(".", ","))
            elif c == "tag":
                row.append(TAG_LABELS.get(v, v))
            elif isinstance(v, float):
                row.append(_fmt(v))
            else:
                row.append("–" if pd.isna(v) else str(v))
        rows.append(row)
    return header, rows, ["left", "left"] + ["right"] * (len(header) - 2)


FIG_RE = re.compile(
    r"^(?P<delta>delta_)?(?P<kind>curvas|real_vs_previsto|residuos)_"
    r"(?:top(?P<rank>\d)_)?(?P<tag>[a-z_]+?)_(?P<model>SimpleRNN|GRU|LSTM)"
    r"_w(?P<w>\d+)_u(?P<u>\d+)_L(?P<l>\d+)$"
)


def figure_entries(fig_dir: Path):
    """Lista (arquivo, legenda) ordenada: top-N, log-retorno e comparação principal."""
    entries = []
    for f in sorted(fig_dir.glob("*.png")):
        m = FIG_RE.match(f.stem)
        if not m:
            continue
        d = m.groupdict()
        camadas = int(d["l"])
        desc = (f"{d['model']}, janela de {d['w']} semanas, {d['u']} unidades, "
                f"{camadas} camada{'s' if camadas > 1 else ''}")
        if d["delta"]:
            group, order = "log-retorno, melhor busca Optuna", 1
            title = f"{FIG_KINDS[d['kind']]} – {desc} ({group})"
        elif d["rank"]:
            order = 0
            title = f"{FIG_KINDS[d['kind']]} – {d['rank']}ª melhor configuração: {desc}"
        else:
            order = 2
            title = f"{FIG_KINDS[d['kind']]} – {desc} ({TAG_LABELS.get(d['tag'], d['tag'])})"
        entries.append(((order, d["rank"] or "9", d["model"], list(FIG_KINDS).index(d["kind"])), f, title, d))

    top_suffixes = {(e[3]["tag"], e[3]["model"], e[3]["w"], e[3]["u"], e[3]["l"])
                    for e in entries if e[3]["rank"]}
    kept = [e for e in entries
            if e[3]["rank"] or e[3]["delta"]
            or (e[3]["tag"], e[3]["model"], e[3]["w"], e[3]["u"], e[3]["l"]) not in top_suffixes]
    kept.sort(key=lambda e: e[0])
    return [(e[1], e[2]) for e in kept]


# --------------------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------------------
def build_report(md_path: Path) -> Report:
    blocks, front = parse_markdown(md_path.read_text(encoding="utf-8"))
    title = next(b[1] for b in blocks if b[0] == "title")
    authors = [a.strip() for a in re.split(r",| e ", plain_text(front.get("Integrantes do grupo", "")))
               if a.strip()]
    professor = f"Prof. Dr. {plain_text(front['Prof. Dr.'])}" if "Prof. Dr." in front else ""
    link = plain_text(front.get("Link do repositório GitHub", ""))

    rep = Report()
    rep.cover(title, authors)
    rep.title_page(title, authors, professor, link)
    rep.toc()
    rep.start_textual_section()

    current_heading = ""
    for b in blocks:
        kind = b[0]
        if kind == "title":
            continue
        if kind == "h1":
            rep.heading(1, b[1])
            current_heading = plain_text(re.sub(r"^\d+\.\s*", "", b[1]))
        elif kind == "h2":
            rep.heading(2, b[1])
            current_heading = plain_text(b[1])
        elif kind == "para":
            rep.para(b[1])
        elif kind == "bullets":
            rep.bullets(b[1])
        elif kind == "table":
            _, header, rows, aligns = b
            key = "|".join(plain_text(h) for h in header[:4])
            rep.table(header, rows, aligns, TABLE_CAPTIONS.get(key, current_heading))

    rep.references()

    # Apêndice A — tabelas completas (seção paisagem para caber as 14 colunas)
    rep._section(landscape=True)
    rep.unnumbered_heading("Apêndice A – Tabelas completas de experimentos", new_page=False)
    header, rows, aligns = experiments_table(config.EXPERIMENTS_CSV)
    rep.table(header, rows, aligns, "Experimentos com a formulação em nível de preço",
              font_size=8, own_section=False)
    delta_csv = config.RESULTS_DIR / "experiments_delta.csv"
    if delta_csv.exists():
        header, rows, aligns = experiments_table(delta_csv)
        rep.table(header, rows, aligns, "Experimentos com a formulação em log-retornos",
                  font_size=8, own_section=False)

    # Apêndice B — figuras
    figures = figure_entries(config.FIGURES_DIR)
    if figures:
        rep._section(landscape=False)
        rep.unnumbered_heading("Apêndice B – Figuras dos experimentos", new_page=False)
        for path, caption in figures:
            rep.figure(path, caption)
    return rep


def convert_to_doc(docx_path: Path) -> Path | None:
    """Converte .docx em .doc (Word 97-2003) com LibreOffice; o textutil do macOS perde layout e imagens."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("[generate_report] LibreOffice não encontrado: apenas o .docx foi gerado.")
        return None
    cmd = [soffice, "--headless", "--convert-to", "doc:MS Word 97", "--outdir", str(docx_path.parent), str(docx_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return docx_path.with_suffix(".doc")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--md", default=str(config.REPORT_DIR / "resultados.md"))
    parser.add_argument("--out", default=str(config.REPORT_DIR / "relatorio_resultados_abnt"))
    args = parser.parse_args()

    out = Path(args.out).with_suffix(".docx")
    rep = build_report(Path(args.md))
    rep.save(out)
    print(f"[generate_report] {out}")
    doc = convert_to_doc(out)
    if doc:
        print(f"[generate_report] {doc}")


if __name__ == "__main__":
    main()
