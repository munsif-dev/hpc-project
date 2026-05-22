#!/usr/bin/env python3
"""Generate the report workflow diagrams and a matching draw.io source file."""

from __future__ import annotations

import html
import math
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_DIR = ROOT / "docs" / "diagrams"
PNG_DIR = DIAGRAM_DIR / "png"

WIDE = (2000, 760)
TALL = (2000, 1180)
TALLER = (2100, 1260)

COLORS = {
    "bg": "#f8fafc",
    "title": "#0f172a",
    "text": "#172033",
    "muted": "#475569",
    "line": "#334155",
    "blue": "#dbeafe",
    "blue_edge": "#2563eb",
    "green": "#dcfce7",
    "green_edge": "#16a34a",
    "amber": "#fef3c7",
    "amber_edge": "#d97706",
    "purple": "#ede9fe",
    "purple_edge": "#7c3aed",
    "rose": "#ffe4e6",
    "rose_edge": "#e11d48",
    "slate": "#e2e8f0",
    "slate_edge": "#475569",
    "white": "#ffffff",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"):
        path = Path(base) / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(58, True)
FONT_BOX = font(46, True)
FONT_SMALL = font(40)
FONT_TINY = font(34)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_to_width(
    draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, fnt)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    box: tuple[int, int, int, int],
    fnt: ImageFont.ImageFont,
    fill: str = COLORS["text"],
    line_gap: int = 9,
) -> None:
    x1, y1, x2, y2 = box
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total) // 2
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        draw.text((x1 + ((x2 - x1) - w) // 2, y), line, font=fnt, fill=fill)
        y += h + line_gap


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    label: str,
    fill: str,
    outline: str,
    *,
    fnt: ImageFont.ImageFont = FONT_BOX,
    radius: int = 26,
    width: int = 5,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    pad = 28
    lines: list[str] = []
    for para in label.split("\n"):
        lines.extend(wrap_to_width(draw, para, fnt, xy[2] - xy[0] - 2 * pad))
    draw_centered_lines(draw, lines, (xy[0] + pad, xy[1] + pad, xy[2] - pad, xy[3] - pad), fnt)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str = COLORS["line"],
    width: int = 6,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 23
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    pts = [
        end,
        (int(end[0] + math.cos(a1) * size), int(end[1] + math.sin(a1) * size)),
        (int(end[0] + math.cos(a2) * size), int(end[1] + math.sin(a2) * size)),
    ]
    draw.polygon(pts, fill=fill)


def connector(draw: ImageDraw.ImageDraw, src: tuple[int, int, int, int], dst: tuple[int, int, int, int]) -> None:
    arrow(draw, (src[2], (src[1] + src[3]) // 2), (dst[0], (dst[1] + dst[3]) // 2))


def base(size: tuple[int, int], title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, COLORS["bg"])
    draw = ImageDraw.Draw(img)
    tw, th = text_size(draw, title, FONT_TITLE)
    draw.text(((size[0] - tw) // 2, 36), title, font=FONT_TITLE, fill=COLORS["title"])
    return img, draw


def save(img: Image.Image, name: str) -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_DIR / name, optimize=True)


def diagram_preprocess() -> None:
    img, draw = base(WIDE, "CB513 Preprocessing")
    boxes = []
    x, y, bw, bh, gap = 62, 235, 290, 275, 35
    labels = [
        "1 Raw CB513 .all files",
        "2 Parse residues and 8-state labels",
        "3 Map DSSP to Q3 H/G/I->H B/E->E else C",
        "4 Build 13-residue windows 260 features",
        "5 Fixed 70/15/15 split seed 42",
        "6 Binary tensors for all variants",
    ]
    for i, label in enumerate(labels):
        boxes.append((x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh))
        rounded_box(draw, boxes[-1], label, COLORS["blue"], COLORS["blue_edge"], fnt=FONT_SMALL)
        if i:
            connector(draw, boxes[i - 1], boxes[i])
    save(img, "01_preprocessing_pipeline.png")


def diagram_serial() -> None:
    img, draw = base(WIDE, "Serial Mini-batch SGD")
    boxes = []
    x, y, bw, bh, gap = 70, 235, 330, 275, 40
    labels = [
        "1 Shuffle mini-batch indices",
        "2 Forward pass compute class scores",
        "3 Backprop compute dW and db",
        "4 SGD update W <- W - lr*dW/B",
        "5 Validate Q3 and continue",
    ]
    colors = [COLORS["slate"], COLORS["blue"], COLORS["amber"], COLORS["green"], COLORS["purple"]]
    edges = [
        COLORS["slate_edge"],
        COLORS["blue_edge"],
        COLORS["amber_edge"],
        COLORS["green_edge"],
        COLORS["purple_edge"],
    ]
    for i, label in enumerate(labels):
        boxes.append((x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh))
        rounded_box(draw, boxes[-1], label, colors[i], edges[i], fnt=FONT_SMALL)
        if i:
            connector(draw, boxes[i - 1], boxes[i])
    save(img, "02_serial_training_loop.png")


def diagram_openmp() -> None:
    img, draw = base(TALL, "OpenMP Shared-memory Data Parallelism")
    b1 = (200, 160, 670, 300)
    b2 = (770, 160, 1240, 300)
    rounded_box(draw, b1, "1 Shared weights W read-only in batch", COLORS["slate"], COLORS["slate_edge"], fnt=FONT_SMALL)
    rounded_box(draw, b2, "2 Static split of mini-batch across threads", COLORS["blue"], COLORS["blue_edge"], fnt=FONT_SMALL)
    connector(draw, b1, b2)

    threads = [
        (190, 475, 610, 670, "3a Thread 0 forward/backward private dW0"),
        (790, 475, 1210, 670, "3b Thread 1 forward/backward private dW1"),
        (1390, 475, 1810, 670, "3c Thread N forward/backward private dWN"),
    ]
    for x1, y1, x2, y2, label in threads:
        rounded_box(draw, (x1, y1, x2, y2), label, COLORS["green"], COLORS["green_edge"], fnt=FONT_TINY)
        arrow(draw, ((b2[0] + b2[2]) // 2, b2[3]), ((x1 + x2) // 2, y1))

    red = (600, 805, 1400, 955)
    upd = (680, 1010, 1320, 1130)
    rounded_box(draw, red, "4 Reduce thread gradients: dW = sum(dWi), db = sum(dbi)", COLORS["amber"], COLORS["amber_edge"], fnt=FONT_SMALL)
    rounded_box(draw, upd, "5 Master applies one SGD update to W and b", COLORS["purple"], COLORS["purple_edge"], fnt=FONT_SMALL)
    for t in threads:
        arrow(draw, ((t[0] + t[2]) // 2, t[3]), ((red[0] + red[2]) // 2, red[1]))
    arrow(draw, ((red[0] + red[2]) // 2, red[3]), ((upd[0] + upd[2]) // 2, upd[1]))
    save(img, "03_openmp_shared_memory.png")


def diagram_pthreads() -> None:
    img, draw = base(TALL, "POSIX Threads Master / Worker Pool")
    b1 = (160, 160, 620, 300)
    b2 = (740, 160, 1240, 300)
    b3 = (1360, 160, 1840, 300)
    rounded_box(draw, b1, "1 Master publishes mini-batch and W", COLORS["slate"], COLORS["slate_edge"], fnt=FONT_SMALL)
    rounded_box(draw, b2, "2 bar_start releases persistent workers", COLORS["blue"], COLORS["blue_edge"], fnt=FONT_SMALL)
    rounded_box(draw, b3, "3 Workers own fixed batch slices", COLORS["green"], COLORS["green_edge"], fnt=FONT_SMALL)
    connector(draw, b1, b2)
    connector(draw, b2, b3)

    workers = [
        (190, 475, 610, 670, "3a Worker 0 local dW0"),
        (790, 475, 1210, 670, "3b Worker 1 local dW1"),
        (1390, 475, 1810, 670, "3c Worker N local dWN"),
    ]
    for x1, y1, x2, y2, label in workers:
        rounded_box(draw, (x1, y1, x2, y2), label, COLORS["green"], COLORS["green_edge"], fnt=FONT_TINY)
        arrow(draw, ((b3[0] + b3[2]) // 2, b3[3]), ((x1 + x2) // 2, y1))

    done = (480, 805, 950, 955)
    upd = (1050, 805, 1520, 955)
    rounded_box(draw, done, "4 bar_done waits until every slice finishes", COLORS["amber"], COLORS["amber_edge"], fnt=FONT_SMALL)
    rounded_box(draw, upd, "5 Master reduces gradients and updates W", COLORS["purple"], COLORS["purple_edge"], fnt=FONT_SMALL)
    for w in workers:
        arrow(draw, ((w[0] + w[2]) // 2, w[3]), ((done[0] + done[2]) // 2, done[1]))
    connector(draw, done, upd)
    save(img, "04_pthreads_master_worker.png")


def diagram_mpi() -> None:
    img, draw = base(TALL, "MPI Distributed-memory Training")
    ranks = [
        (100, 185, 590, 405, "1a Rank 0 private W and batch slice"),
        (755, 185, 1245, 405, "1b Rank 1 private W and batch slice"),
        (1410, 185, 1900, 405, "1c Rank R-1 private W and batch slice"),
    ]
    grads = []
    for x1, y1, x2, y2, label in ranks:
        rounded_box(draw, (x1, y1, x2, y2), label, COLORS["blue"], COLORS["blue_edge"], fnt=FONT_TINY)
        g = (x1, 535, x2, 710)
        grads.append(g)
        rounded_box(draw, g, "2 Local forward/backward builds rank gradient", COLORS["green"], COLORS["green_edge"], fnt=FONT_TINY)
        arrow(draw, ((x1 + x2) // 2, y2), ((x1 + x2) // 2, g[1]))
    allr = (560, 825, 1440, 980)
    upd = (590, 1030, 1410, 1150)
    rounded_box(draw, allr, "3-4 Pack gradients, MPI_Allreduce(SUM) across ranks", COLORS["amber"], COLORS["amber_edge"], fnt=FONT_SMALL)
    rounded_box(draw, upd, "5 Every rank applies the same SGD update", COLORS["purple"], COLORS["purple_edge"], fnt=FONT_SMALL)
    for g in grads:
        arrow(draw, ((g[0] + g[2]) // 2, g[3]), ((allr[0] + allr[2]) // 2, allr[1]))
    arrow(draw, ((allr[0] + allr[2]) // 2, allr[3]), ((upd[0] + upd[2]) // 2, upd[1]))
    save(img, "05_mpi_distributed_memory.png")


def diagram_hybrid() -> None:
    img, draw = base(TALLER, "Hybrid MPI + OpenMP")
    panels = [
        (90, 160, 990, 660, "1 Rank 0 owns a batch slice"),
        (1110, 160, 2010, 660, "1 Rank 1 owns a batch slice"),
    ]
    rank_reduces = []
    for x1, y1, x2, y2, title in panels:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=28, fill=COLORS["white"], outline=COLORS["blue_edge"], width=6)
        draw_centered_lines(draw, wrap_to_width(draw, title, FONT_SMALL, x2 - x1 - 80), (x1 + 40, y1 + 28, x2 - 40, y1 + 110), FONT_SMALL)
        thread_boxes = [
            (x1 + 60, y1 + 185, x1 + 320, y1 + 390, "2a Thread 0 local grad"),
            (x1 + 320, y1 + 185, x1 + 580, y1 + 390, "2b Thread 1 local grad"),
            (x1 + 580, y1 + 185, x1 + 840, y1 + 390, "2c Thread T local grad"),
        ]
        for tb in thread_boxes:
            rounded_box(draw, tb[:4], tb[4], COLORS["green"], COLORS["green_edge"], fnt=FONT_TINY)
        rr = (x1 + 180, y1 + 430, x2 - 180, y1 + 585)
        rank_reduces.append(rr)
        rounded_box(draw, rr, "3 Intra-rank reduce to one rank gradient", COLORS["amber"], COLORS["amber_edge"], fnt=FONT_TINY)
        for tb in thread_boxes:
            arrow(draw, ((tb[0] + tb[2]) // 2, tb[3]), ((rr[0] + rr[2]) // 2, rr[1]))

    allr = (530, 790, 1570, 960)
    upd = (620, 1045, 1480, 1200)
    rounded_box(draw, allr, "4 MPI_Allreduce combines rank gradients", COLORS["rose"], COLORS["rose_edge"], fnt=FONT_SMALL)
    rounded_box(draw, upd, "5 All ranks apply identical SGD update", COLORS["purple"], COLORS["purple_edge"], fnt=FONT_SMALL)
    for rr in rank_reduces:
        arrow(draw, ((rr[0] + rr[2]) // 2, rr[3]), ((allr[0] + allr[2]) // 2, allr[1]))
    arrow(draw, ((allr[0] + allr[2]) // 2, allr[3]), ((upd[0] + upd[2]) // 2, upd[1]))
    save(img, "06_hybrid_mpi_openmp.png")


def diagram_cuda() -> None:
    img, draw = base(TALLER, "CUDA GPU Training Workflow")
    host = (90, 160, 2010, 350)
    gpu = (90, 430, 2010, 1170)
    draw.rounded_rectangle(host, radius=30, fill="#eff6ff", outline=COLORS["blue_edge"], width=6)
    draw.rounded_rectangle(gpu, radius=30, fill="#f0fdf4", outline=COLORS["green_edge"], width=6)
    draw.text((130, 195), "CPU host", font=FONT_BOX, fill=COLORS["title"])
    draw.text((130, 465), "GPU device", font=FONT_BOX, fill=COLORS["title"])

    hb = (660, 200, 1430, 315)
    rounded_box(draw, hb, "1 Shuffle batch indices and launch kernels", COLORS["white"], COLORS["blue_edge"], fnt=FONT_TINY)

    boxes = [
        (150, 570, 470, 760, "2 Data and W stay resident in GPU memory"),
        (560, 570, 880, 760, "3 Parallel gather builds column-major batch"),
        (970, 570, 1290, 760, "4 Forward GEMMs plus ReLU kernels"),
        (1380, 570, 1700, 760, "5 Softmax loss computes dZ in parallel"),
        (970, 880, 1290, 1070, "6 Backward GEMMs update W in place"),
        (1380, 880, 1700, 1070, "7 Copy W to CPU only for Q3 evaluation"),
    ]
    for i, (x1, y1, x2, y2, label) in enumerate(boxes):
        fill = COLORS["green"] if i < 4 else COLORS["amber"]
        edge = COLORS["green_edge"] if i < 4 else COLORS["amber_edge"]
        rounded_box(draw, (x1, y1, x2, y2), label, fill, edge, fnt=FONT_TINY)
    arrow(draw, ((hb[0] + hb[2]) // 2, hb[3]), ((boxes[1][0] + boxes[1][2]) // 2, boxes[1][1]))
    for a, b in zip(boxes[:4], boxes[1:4]):
        connector(draw, a[:4], b[:4])
    arrow(draw, ((boxes[3][0] + boxes[3][2]) // 2, boxes[3][3]), ((boxes[4][0] + boxes[4][2]) // 2, boxes[4][1]))
    connector(draw, boxes[4][:4], boxes[5][:4])
    arrow(draw, (boxes[4][0], (boxes[4][1] + boxes[4][3]) // 2), ((boxes[0][0] + boxes[0][2]) // 2, boxes[0][3]))
    save(img, "07_cuda_gpu_workflow.png")


def make_drawio() -> None:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-05-21T00:00:00.000Z",
            "agent": "Codex",
            "version": "24.0.0",
            "type": "device",
        },
    )

    def page(page_name: str, width: int = 1600, height: int = 900) -> tuple[ET.Element, ET.Element]:
        diagram = ET.SubElement(mxfile, "diagram", {"id": page_name.lower().replace(" ", "-"), "name": page_name})
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": str(width),
                "dy": str(height),
                "grid": "1",
                "gridSize": "10",
                "page": "1",
                "pageWidth": str(width),
                "pageHeight": str(height),
            },
        )
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        return root, diagram

    def node(
        root: ET.Element,
        node_id: str,
        label: str,
        x: int,
        y: int,
        w: int,
        h: int,
        fill: str = "#dbeafe",
        stroke: str = "#2563eb",
    ) -> None:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": node_id,
                "value": html.escape(label),
                "style": f"rounded=1;whiteSpace=wrap;html=1;fontSize=14;strokeWidth=2;fillColor={fill};strokeColor={stroke};",
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def edge(root: ET.Element, edge_id: str, source: str, target: str) -> None:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": edge_id,
                "style": "endArrow=block;html=1;rounded=0;fontSize=14;strokeWidth=2;",
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    root, _ = page("Preprocessing")
    ids = ["raw", "parse", "q3", "win", "split", "bin"]
    labels = [
        "1 Raw CB513 .all files",
        "2 Parse residues and 8-state labels",
        "3 Collapse DSSP to Q3",
        "4 Build 13-residue windows",
        "5 Fixed train/val/test split",
        "6 Binary tensors for all variants",
    ]
    for i, (node_id, label) in enumerate(zip(ids, labels)):
        node(root, node_id, label, 40 + i * 250, 260, 210, 120)
        if i:
            edge(root, f"e{i}", ids[i - 1], node_id)

    root, _ = page("Serial SGD")
    ids = ["shuffle", "forward", "backprop", "update", "eval"]
    labels = [
        "1 Shuffle mini-batch indices",
        "2 Forward pass",
        "3 Backprop gradients",
        "4 SGD update W <- W - lr*dW/B",
        "5 Validate Q3",
    ]
    fills = ["#e2e8f0", "#dbeafe", "#fef3c7", "#dcfce7", "#ede9fe"]
    strokes = ["#475569", "#2563eb", "#d97706", "#16a34a", "#7c3aed"]
    for i, (node_id, label) in enumerate(zip(ids, labels)):
        node(root, node_id, label, 110 + i * 290, 260, 240, 120, fills[i], strokes[i])
        if i:
            edge(root, f"e{i}", ids[i - 1], node_id)

    root, _ = page("OpenMP", height=1050)
    node(root, "w", "1 Shared weights read-only", 220, 120, 300, 110, "#e2e8f0", "#475569")
    node(root, "split", "2 Static mini-batch split across threads", 660, 120, 360, 110)
    edge(root, "e1", "w", "split")
    for i, x in enumerate([180, 620, 1060]):
        node(root, f"t{i}", f"3{chr(97+i)} Thread {i if i < 2 else 'N'} private gradient", x, 390, 300, 130, "#dcfce7", "#16a34a")
        edge(root, f"es{i}", "split", f"t{i}")
        edge(root, f"er{i}", f"t{i}", "reduce")
    node(root, "reduce", "4 Reduce thread gradients", 530, 680, 460, 120, "#fef3c7", "#d97706")
    node(root, "update", "5 Master applies one SGD update", 570, 870, 380, 100, "#ede9fe", "#7c3aed")
    edge(root, "eu", "reduce", "update")

    root, _ = page("Pthreads", height=1050)
    ids = ["master", "start", "assign"]
    labels = ["1 Master publishes batch", "2 bar_start releases workers", "3 Workers process fixed slices"]
    for i, (node_id, label) in enumerate(zip(ids, labels)):
        node(root, node_id, label, 120 + i * 430, 120, 330, 110, ["#e2e8f0", "#dbeafe", "#dcfce7"][i], ["#475569", "#2563eb", "#16a34a"][i])
        if i:
            edge(root, f"e{i}", ids[i - 1], node_id)
    for i, x in enumerate([180, 620, 1060]):
        node(root, f"wk{i}", f"3{chr(97+i)} Worker {i if i < 2 else 'N'} local dW", x, 390, 300, 130, "#dcfce7", "#16a34a")
        edge(root, f"ew{i}", "assign", f"wk{i}")
        edge(root, f"ed{i}", f"wk{i}", "done")
    node(root, "done", "4 bar_done waits for all slices", 420, 700, 360, 120, "#fef3c7", "#d97706")
    node(root, "update", "5 Master reduces gradients and updates W", 850, 700, 430, 120, "#ede9fe", "#7c3aed")
    edge(root, "eu", "done", "update")

    root, _ = page("MPI", height=1050)
    for i, x in enumerate([120, 620, 1120]):
        node(root, f"r{i}", f"1{chr(97+i)} Rank {i if i < 2 else 'R-1'} private W and batch slice", x, 140, 340, 130)
        node(root, f"g{i}", "2 Local forward/backward rank gradient", x, 400, 340, 130, "#dcfce7", "#16a34a")
        edge(root, f"eg{i}", f"r{i}", f"g{i}")
        edge(root, f"ea{i}", f"g{i}", "allreduce")
    node(root, "allreduce", "3-4 Pack gradients, MPI_Allreduce(SUM)", 500, 690, 600, 120, "#fef3c7", "#d97706")
    node(root, "update", "5 Same SGD update on every rank", 560, 880, 480, 100, "#ede9fe", "#7c3aed")
    edge(root, "eu", "allreduce", "update")

    root, _ = page("Hybrid", width=1800, height=1100)
    for rank, x0 in enumerate([90, 950]):
        node(root, f"rank{rank}", f"1 Rank {rank} owns a batch slice", x0, 130, 760, 100, "#ffffff", "#2563eb")
        for t, x in enumerate([x0 + 60, x0 + 270, x0 + 480]):
            node(root, f"r{rank}t{t}", f"2{chr(97+t)} Thread {t if t < 2 else 'T'} local grad", x, 330, 190, 120, "#dcfce7", "#16a34a")
            edge(root, f"et{rank}{t}", f"rank{rank}", f"r{rank}t{t}")
            edge(root, f"err{rank}{t}", f"r{rank}t{t}", f"rr{rank}")
        node(root, f"rr{rank}", "3 Intra-rank reduce", x0 + 220, 550, 320, 100, "#fef3c7", "#d97706")
        edge(root, f"ea{rank}", f"rr{rank}", "allreduce")
    node(root, "allreduce", "4 MPI_Allreduce combines rank gradients", 530, 760, 740, 120, "#ffe4e6", "#e11d48")
    node(root, "update", "5 Identical SGD update on all ranks", 620, 950, 560, 100, "#ede9fe", "#7c3aed")
    edge(root, "eu", "allreduce", "update")

    root, _ = page("CUDA", width=1800, height=1100)
    node(root, "host", "CPU host: 1 Shuffle batch indices and launch kernels", 160, 100, 1480, 110, "#eff6ff", "#2563eb")
    cuda_nodes = [
        ("resident", "2 Data and W stay resident on GPU", 140, 430),
        ("gather", "3 Parallel gather column-major batch", 440, 430),
        ("forward", "4 Forward cuBLAS GEMMs + ReLU kernels", 740, 430),
        ("loss", "5 Softmax loss computes dZ in parallel", 1040, 430),
        ("backward", "6 Backward GEMMs update W in place", 740, 730),
        ("eval", "7 Copy W to CPU only for Q3 evaluation", 1040, 730),
    ]
    for node_id, label, x, y in cuda_nodes:
        fill, stroke = ("#dcfce7", "#16a34a") if node_id not in {"backward", "eval"} else ("#fef3c7", "#d97706")
        node(root, node_id, label, x, y, 240, 130, fill, stroke)
    edge(root, "eh", "host", "gather")
    for source, target in [("resident", "gather"), ("gather", "forward"), ("forward", "loss"), ("loss", "backward"), ("backward", "eval"), ("backward", "resident")]:
        edge(root, f"e{source}{target}", source, target)

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(DIAGRAM_DIR / "hpc_workflows.drawio", encoding="utf-8", xml_declaration=True)


def make_pdf() -> None:
    images = [Image.open(p).convert("RGB") for p in sorted(PNG_DIR.glob("0*.png"))]
    if images:
        images[0].save(DIAGRAM_DIR / "hpc_workflows.pdf", save_all=True, append_images=images[1:])


def main() -> None:
    diagram_preprocess()
    diagram_serial()
    diagram_openmp()
    diagram_pthreads()
    diagram_mpi()
    diagram_hybrid()
    diagram_cuda()
    make_drawio()
    make_pdf()


if __name__ == "__main__":
    main()
