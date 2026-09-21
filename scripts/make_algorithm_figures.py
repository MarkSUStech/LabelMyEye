"""Generate all figures for docs/DR算法说明.md (Chinese labels via msyh font).

Usage: python scripts/make_algorithm_figures.py
Outputs to docs/figures/*.png
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labelmyeye.dr_geometry import (  # noqa: E402
    DRParams, elliptic_radius, nasal_depth_floor, oval_polygon,
    oval_radius, posterior_pole_ellipse, tangent_ellipse_two_circles)
from calibrate_from_jsons import fit_circle_rms  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
CAL = os.path.join(ROOT, "scripts", "calibration_data")
PARAMS = DRParams().validated()


def font(size: int):
    for cand in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                 "C:/Windows/Fonts/simsun.ttc"):
        if os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    return ImageFont.load_default()


F = lambda s: font(s)


def synthetic_background(w=1240, h=1129):
    """Deterministic schematic fundus background (no patient data)."""
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, fr = 590.0, 555.0, 565.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(1 - dist / fr, 0, 1) ** 0.75
    img = np.zeros((h, w, 3))
    img[..., 0] = 25 + 190 * t
    img[..., 1] = 18 + 115 * t
    img[..., 2] = 14 + 62 * t
    dxx, dyy = xx - 409.0, yy - 592.0
    dd = np.sqrt(dxx ** 2 + dyy ** 2)
    od = np.clip(1 - dd / (90.2 * 1.6), 0, 1) ** 1.6
    img[..., 0] += 95 * od
    img[..., 1] += 75 * od
    img[..., 2] += 40 * od
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


def load_sample(name="0003_3"):
    data = json.load(open(os.path.join(CAL, name + ".json"), encoding="utf-8"))
    S = {s["label"]: np.array(s["points"], float) for s in data["shapes"]}
    img = synthetic_background(*Image.open(os.path.join(CAL, name + ".png")).size)
    (cd, Rd), _ = fit_circle_rms(S["optic"])
    (cm, Rm), _ = fit_circle_rms(S["macular"])
    return img, S, (cd, Rd), (cm, Rm)


def draw_ellipse(d, c, a, b, theta, col, width=3, dash=False):
    ts = np.linspace(0, 2 * math.pi, 240)
    xs = c[0] + a * np.cos(ts) * math.cos(theta) - b * np.sin(ts) * math.sin(theta)
    ys = c[1] + a * np.cos(ts) * math.sin(theta) + b * np.sin(ts) * math.cos(theta)
    if dash:
        pts = list(zip(xs, ys))
        for i in range(0, len(pts) - 3, 6):
            d.line([pts[i], pts[i + 1], pts[i + 2]], fill=col, width=width)
    else:
        d.line(list(zip(xs, ys)), fill=col, width=width)


def draw_circle(d, c, r, col, width=3):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=col, width=width)


def arrow(d, p0, p1, col, width=3, text=None, font=None, text_off=(0, 0)):
    d.line([p0, p1], fill=col, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (1, -1):
        hx = p1[0] - 10 * math.cos(ang - s * 0.45)
        hy = p1[1] - 10 * math.sin(ang - s * 0.45)
        d.line([p1, (hx, hy)], fill=col, width=width)
    if text and font:
        mx, my = (p0[0] + p1[0]) / 2 + text_off[0], (p0[1] + p1[1]) / 2 + text_off[1]
        d.text((mx, my), text, fill=col, font=font)


# ---------------------------------------------------------------- fig1 输入
def fig1():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    d = ImageDraw.Draw(img)
    draw_circle(d, cd, Rd, (255, 70, 70), 5)
    draw_circle(d, cm, Rm, (70, 130, 255), 5)
    F30 = F(34)
    d.text((cd[0] - 40, cd[1] + Rd + 18), "视盘圆 (Cd, Rd)", fill=(255, 120, 120), font=F30)
    d.text((cm[0] - 60, cm[1] + Rm + 18), "黄斑圆 (Cm, Rm)", fill=(120, 170, 255), font=F30)
    arrow(d, (cd[0] + Rd * 0.3, cd[1] - Rd * 0.3), (cm[0] - Rm * 0.3, cm[1] - Rm * 0.3),
          (255, 255, 255), 3)
    d.text(((cd[0] + cm[0]) / 2 - 130, (cd[1] + cm[1]) / 2 - 90), "轴: 视盘中心→黄斑中心",
           fill=(255, 255, 255), font=F(28))
    d.text((30, 20), "输入: 分割得到的视盘圆与黄斑圆（点击→SAM→拟合圆）", fill=(255, 255, 255), font=F(30))
    img.save(os.path.join(OUT, "fig1_input.png"))


# ------------------------------------------------------- fig2 绿椭圆(相切)
def fig2():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    ell = tangent_ellipse_two_circles(cd[:2], Rd, cm[:2], Rm)
    d = ImageDraw.Draw(img)
    draw_circle(d, cd, Rd, (255, 70, 70), 5)
    draw_circle(d, cm, Rm, (70, 130, 255), 5)
    draw_ellipse(d, ell["c"], ell["a"], ell["b"], ell["theta"], (70, 200, 120), 5)
    # 相切点标记: 鼻侧极点 & 颞侧极点
    u = np.array([(cm[0] - cd[0]), (cm[1] - cd[1])])
    u = u / math.hypot(*u)
    p1 = (cd[0] - Rd * u[0], cd[1] - Rd * u[1])
    p2 = (cm[0] + Rm * u[0], cm[1] + Rm * u[1])
    for pt, txt in ((p1, "相切点①(视盘鼻侧缘)"), (p2, "相切点②(黄斑颞侧缘)")):
        d.ellipse([pt[0] - 8, pt[1] - 8, pt[0] + 8, pt[1] + 8], fill=(255, 230, 80))
    d.text((p1[0] - 40, p1[1] + 60), "相切点①", fill=(255, 230, 80), font=F(28))
    d.text((p2[0] - 30, p2[1] - 70), "相切点②", fill=(255, 230, 80), font=F(28))
    d.text((30, 20), "macular_area(绿): 与两圆同时相切、两端顶点落在相切点①②的椭圆",
           fill=(255, 255, 255), font=F(30))
    d.text((30, 62), "中心 = ①②中点, 长半轴 a = |①②|/2, 短半轴 b = 恰好包住两圆的最小值",
           fill=(220, 220, 220), font=F(26))
    img.save(os.path.join(OUT, "fig2_tangent.png"))


# ---------------------------------------------------- fig3 后极部(边缘外扩)
def fig3():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    p = PARAMS
    ell = tangent_ellipse_two_circles(cd[:2], Rd, cm[:2], Rm)
    pp = posterior_pole_ellipse(cd[:2], Rd, cm[:2], Rm, p)
    d = ImageDraw.Draw(img)
    draw_circle(d, cd, Rd, (255, 70, 70), 4)
    draw_circle(d, cm, Rm, (70, 130, 255), 4)
    draw_ellipse(d, ell["c"], ell["a"], ell["b"], ell["theta"], (70, 200, 120), 4)
    from labelmyeye.dr_geometry import oval_polygon
    pp_pts = oval_polygon(pp, 180)
    d.line([tuple(q) for q in pp_pts] + [tuple(pp_pts[0])], fill=(250, 160, 60), width=5)
    u = np.array([(cm[0] - cd[0]), (cm[1] - cd[1])])
    u = u / math.hypot(*u)
    n = np.array([-u[1], u[0]])
    pd = 2 * Rd
    # 颞侧 2PD
    g_t0 = ell["c"] + ell["a"] * u
    g_t1 = pp["c"] + pp["a_t"] * u
    arrow(d, tuple(g_t0), tuple(g_t1), (255, 230, 80), 4, "2PD(颞侧)",
          F(28), (30, -20))
    # 鼻侧 1PD
    g_n0 = ell["c"] - ell["a_n"] * u if False else ell["c"] - ell["a"] * u
    g_n1 = pp["c"] - pp["a_n"] * u
    arrow(d, tuple(g_n0), tuple(g_n1), (255, 230, 80), 4, "1PD(鼻侧)",
          F(28), (-160, -20))
    # 上下 2PD
    for s in (1, -1):
        v0 = ell["c"] + s * ell["b"] * n
        v1 = pp["c"] + s * pp["b"] * n
        arrow(d, tuple(v0), tuple(v1), (255, 230, 80), 4,
              "2PD" if s > 0 else "2PD", F(28),
              (10 if s > 0 else 10, -46 if s > 0 else 10))
    d.text((30, 20), "posterior_pole_boundary(橙): 绿椭圆边缘向外扩 — 颞侧2PD · 鼻侧1PD · 上下2PD",
           fill=(255, 255, 255), font=F(30))
    img.save(os.path.join(OUT, "fig3_pp.png"))


# --------------------------------------------- fig4 central_inner 构造分解
def fig4():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    p = PARAMS
    pp = posterior_pole_ellipse(cd[:2], Rd, cm[:2], Rm, p)
    axv = (cm[0] - cd[0], cm[1] - cd[1])
    axis_ang = math.atan2(axv[1], axv[0])
    E = np.array(pp["c"])
    n = 720
    phis = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rel = (phis - axis_ang + math.pi) % (2 * math.pi) - math.pi
    delta = rel + (axis_ang - pp["theta"])
    r_pp = np.asarray(oval_radius(pp, delta))
    s = math.radians(p.ci_width)

    def bump(center):
        x = (rel - center + math.pi) % (2 * math.pi) - math.pi
        return np.exp(-0.5 * (x / s) ** 2)

    depth_sup = p.ci_depth_sup * bump(math.radians(p.ci_angle_sup))
    depth_inf = p.ci_depth_inf * bump(math.radians(p.ci_angle_inf))
    depth = np.maximum(depth_sup, depth_inf)
    r_min_both = r_pp * (1.0 - depth)   # min(蓝,绿): 未扣底噪的合并轮廓
    r_sup_only = r_pp * (1.0 - depth_sup)   # 只挖上凹口
    r_inf_only = r_pp * (1.0 - depth_inf)   # 只挖下凹口
    f = E - np.array(cd[:2])
    b_ = np.cos(phis) * f[0] + np.sin(phis) * f[1]
    cc = f @ f - (Rd * (1.0 + 2.0 * p.ci_bubble_pd)) ** 2
    r_bub = np.where(cc >= 0, -b_ + np.sqrt(np.maximum(b_ * b_ - cc, 0)), 0.0)
    # 最终轮廓: 直接取软件 central_inner_polygon 的输出(鼻侧底噪扣除 +
    # 可选气泡都在算法内部), 不在这里重写公式 —— 否则会和真实算法漂移
    r_final = _ci_radial_from_polygon(cd, Rd, cm, Rm, p, E, phis)

    d = ImageDraw.Draw(img)
    pp_pts = np.asarray(oval_polygon(pp, 240))
    dash_pts = [tuple(q) for i, q in enumerate(pp_pts) if i % 4 < 2]
    for i in range(0, len(dash_pts) - 1, 2):
        d.line([dash_pts[i], dash_pts[i + 1]], fill=(250, 160, 60), width=3)

    def poly(rr, col, width, dash=False):
        pts = [(E[0] + r * math.cos(t), E[1] + r * math.sin(t))
               for t, r in zip(phis, rr)]
        if dash:
            for i in range(0, len(pts) - 3, 6):
                d.line([pts[i], pts[i + 1], pts[i + 2]], fill=col, width=width)
        else:
            d.line(pts + [pts[0]], fill=col, width=width)

    poly(r_sup_only, (90, 160, 230), 3)       # 只挖上凹口
    poly(r_inf_only, (60, 200, 130), 3)       # 只挖下凹口
    poly(r_bub, (140, 190, 150), 3)           # 视盘包绕气泡（默认关闭）
    poly(r_min_both, (0, 0, 0), 7, dash=True) # min(蓝,绿), 未扣底噪
    poly(r_final, (170, 110, 255), 6)         # 最终
    draw_circle(d, cd, Rd, (255, 70, 70), 4)
    draw_circle(d, cm, Rm, (70, 130, 255), 4)
    d.text((30, 20), "central_inner(紫) = 后极部卵圆让出上、下凹口(避开血管弓外激光区) + 鼻侧底噪扣除",
           fill=(255, 255, 255), font=F(30))
    legend = [("后极部卵圆(橙虚线): 外极限轮廓", (250, 160, 60)),
              ("只让出上凹口(蓝细线): 上方激光安全区", (90, 160, 230)),
              ("只让出下凹口(绿细线): 下方激光安全区", (60, 200, 130)),
              ("两凹口逐方向取更靠内(黑粗虚线): = min(蓝,绿), 未扣鼻侧底噪, 鼻侧被尾部侵蚀",
               (0, 0, 0)),
              ("视盘包绕气泡(灰绿细线): 可选功能, 默认关闭(此图未启用)", (140, 190, 150)),
              ("最终轮廓(紫粗线): 黑线再做鼻侧底噪扣除, 鼻侧贴回1PD线", (170, 110, 255))]
    for i, (txt, col) in enumerate(legend):
        d.line([(45, 760 + i * 44), (105, 760 + i * 44)], fill=col, width=5)
        d.text((120, 760 + i * 44 - 8), txt, fill=(240, 240, 240), font=F(26))
    img.save(os.path.join(OUT, "fig4_construction.png"))


# ------------------------------------------------------------- fig5 径向剖面
def fig5():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    p = PARAMS
    pp = posterior_pole_ellipse(cd[:2], Rd, cm[:2], Rm, p)
    axv = (cm[0] - cd[0], cm[1] - cd[1])
    axis_ang = math.atan2(axv[1], axv[0])
    E = np.array(pp["c"])
    n = 720
    phis = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rel = (phis - axis_ang + math.pi) % (2 * math.pi) - math.pi
    delta = rel + (axis_ang - pp["theta"])
    r_pp = np.asarray(oval_radius(pp, delta))
    r_final = _ci_radial_from_polygon(cd, Rd, cm, Rm, p, E, phis)

    W, H = 1400, 560
    canvas = Image.new("RGB", (W, H), (38, 36, 42))
    d = ImageDraw.Draw(canvas)
    ox, oy, sx, sy = 110, 80, W - 190, 2.6
    for gy in range(0, 700, 100):
        y = oy + 400 - gy * sy / 2.6 * 0.65
        d.line([(ox, y), (W - 60, y)], fill=(70, 70, 78))
        d.text((30, y - 10), f"{gy} px", fill=(150, 150, 160), font=F(22))
    deg = np.degrees(phis)
    def curve(rr, col, label):
        pts = [(ox + deg_i / 360 * sx, oy + 400 - r_i * sy * 0.25)
               for deg_i, r_i in zip(deg, rr)]
        d.line(pts, fill=col, width=4)
        lx = ox + sx * 0.30
        ly = oy + 400 - rr[int(n * 0.30)] * sy * 0.25
        d.text((lx, ly - 34), label, fill=col, font=F(26))
    curve(r_pp, (250, 160, 60), "R_pp(φ) 后极部卵圆")
    curve(r_final, (170, 110, 255), "r(φ) central_inner 最终轮廓")
    # 凹口中心标注
    for a_deg, txt in ((p.ci_angle_sup % 360, "上凹口中心"),
                       ((p.ci_angle_inf + 360) % 360, "下凹口中心")):
        x = ox + a_deg / 360 * sx
        d.line([(x, oy - 10), (x, oy + 420)], fill=(120, 120, 130), width=2)
        d.text((x - 40, oy - 40), txt, fill=(200, 200, 210), font=F(24))
    for lbl, x in (("颞侧 0°", ox), ("上凹口 +123°", ox + 123 / 360 * sx),
                   ("鼻侧 180°", ox + 0.5 * sx),
                   ("下凹口 −128°", ox + 232 / 360 * sx)):
        d.text((x - 50, oy + 425), lbl, fill=(180, 180, 190), font=F(24))
    d.text((40, 15), "径向轮廓: 从后极部中心 E 看, 每个方向 φ 上的边界距离 r(φ)",
           fill=(240, 240, 240), font=F(30))
    d.text((40, 55), "凹口处 r = R_pp×(1−深度) —— 上下方向让出激光安全区; 其余方向贴住后极部",
           fill=(200, 200, 205), font=F(25))
    canvas.save(os.path.join(OUT, "fig5_profile.png"))


# ------------------------------------------------------------- fig6 标定效果
def fig6():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    p = PARAMS
    ci_model = central_inner_points(cd, Rd, cm, Rm, p)
    d = ImageDraw.Draw(img)
    pts = [tuple(q) for q in S["central_inner"]]
    for x, y in pts:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255))
    d.line([tuple(q) for q in ci_model] + [tuple(ci_model[0])],
           fill=(170, 110, 255), width=5)
    d.text((30, 20), "标定效果(样例 0003_3): 白点=组内手绘 central_inner, 紫线=算法输出",
           fill=(255, 255, 255), font=F(30))
    d.text((30, 62), "14 例平均边界误差 0.31 视盘半径(手绘折线自身抖动约 0.5)", fill=(220, 220, 220), font=F(26))
    img.save(os.path.join(OUT, "fig6_calibration.png"))


# ------------------------------------------------- fig9 鼻侧底噪扣除对比
def _ci_radial_from_polygon(cd, Rd, cm, Rm, p, E, phis):
    """软件 central_inner_polygon() 的输出折算成 r(φ)。

    图上的"最终轮廓"一律由此计算, 保证与 labelmyeye.dr_geometry 的
    真实算法完全一致(含鼻侧底噪扣除与可选气泡), 不在绘图脚本里重写公式。
    """
    from labelmyeye.dr_geometry import central_inner_polygon
    pts = np.asarray(central_inner_polygon(cd, Rd, cm, Rm, p))
    tt = np.arctan2(pts[:, 1] - E[1], pts[:, 0] - E[0]) % (2.0 * math.pi)
    rr = np.hypot(pts[:, 0] - E[0], pts[:, 1] - E[1])
    order = np.argsort(tt)
    return np.interp(phis % (2.0 * math.pi), tt[order], rr[order],
                     period=2.0 * math.pi)


def _ci_curves(cd, Rd, cm, Rm, p, n=720):
    """Radial profiles for the nasal-floor comparison.

    Returns (phis, rel, r_pp, r_off, r_on, E):
      r_off = 无鼻侧底噪扣除, r_on = 扣除后（软件实际输出）.
    """
    pp = posterior_pole_ellipse(cd[:2], Rd, cm[:2], Rm, p)
    axv = (cm[0] - cd[0], cm[1] - cd[1])
    axis_ang = math.atan2(axv[1], axv[0])
    E = np.array(pp["c"])
    phis = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rel = (phis - axis_ang + math.pi) % (2 * math.pi) - math.pi
    delta = rel + (axis_ang - pp["theta"])
    r_pp = np.asarray(oval_radius(pp, delta))
    s = math.radians(p.ci_width)

    def bump(center):
        x = (rel - center + math.pi) % (2 * math.pi) - math.pi
        return np.exp(-0.5 * (x / s) ** 2)

    depth = np.maximum(p.ci_depth_sup * bump(math.radians(p.ci_angle_sup)),
                       p.ci_depth_inf * bump(math.radians(p.ci_angle_inf)))
    r_off = r_pp * (1.0 - depth)
    r_on = _ci_radial_from_polygon(cd, Rd, cm, Rm, p, E, phis)
    return phis, rel, r_pp, r_off, r_on, E


def fig9():
    img, S, (cd, Rd), (cm, Rm) = load_sample()
    p = PARAMS
    phis, rel, r_pp, r_off, r_on, E = _ci_curves(cd, Rd, cm, Rm, p)
    pp = posterior_pole_ellipse(cd[:2], Rd, cm[:2], Rm, p)
    floor_val = nasal_depth_floor(p)
    axv = (cm[0] - cd[0], cm[1] - cd[1])
    u = np.array(axv, dtype=float)
    u = u / math.hypot(*u)
    erosion_px = float(floor_val * pp["a_n"])
    erosion_rd = erosion_px / Rd

    C_OFF, C_ON = (60, 220, 220), (170, 110, 255)

    # ---- 上半：眼底叠加（鼻侧局部） ----
    work = img.copy()
    d = ImageDraw.Draw(work)
    pp_pts = np.asarray(oval_polygon(pp, 240))
    dash_pts = [tuple(q) for i, q in enumerate(pp_pts) if i % 4 < 2]
    for i in range(0, len(dash_pts) - 1, 2):
        d.line([dash_pts[i], dash_pts[i + 1]], fill=(250, 160, 60), width=3)

    def poly(rr, col, width, dash=False):
        pts = [(E[0] + r * math.cos(t), E[1] + r * math.sin(t))
               for t, r in zip(phis, rr)]
        if dash:
            for i in range(0, len(pts) - 3, 6):
                d.line([pts[i], pts[i + 1], pts[i + 2]], fill=col, width=width)
        else:
            d.line(pts + [pts[0]], fill=col, width=width)

    poly(r_off, C_OFF, 5, dash=True)     # 未扣除底噪
    poly(r_on, C_ON, 8)                  # 扣除后（实际输出）
    draw_circle(d, cd, Rd, (255, 70, 70), 4)

    nasal_hi = E - pp["a_n"] * u                     # 鼻极（贴回 1PD 线）
    nasal_lo = E - (pp["a_n"] - erosion_px) * u      # 未扣除时的鼻侧边界
    arrow(d, tuple(nasal_lo), tuple(nasal_hi), (255, 230, 80), 4)

    box = (0, 120, 780, 940)
    cw, ch = box[2] - box[0], box[3] - box[1]
    dh = 600
    k = dh / ch
    crop = work.crop(box).resize((int(cw * k), dh))
    dc = ImageDraw.Draw(crop)
    ax0 = (int((nasal_lo[0] - box[0]) * k) + 14,
           int((nasal_lo[1] - box[1]) * k) + 26)
    tip = (ax0[0] - 8, ax0[1] - 12)
    tx0, ty0 = 48, 424
    tw, th = 392, 42
    dc.rectangle([tx0, ty0, tx0 + tw, ty0 + th], fill=(18, 18, 22))
    dc.text((tx0 + 12, ty0 + 6), f"鼻侧被侵蚀 ≈ {erosion_rd:.1f} 视盘半径",
            fill=(255, 230, 80), font=F(26))
    _ = tip

    # ---- 下半：径向剖面 ----
    W, H = 1400, 1250
    canvas = Image.new("RGB", (W, H), (38, 36, 42))
    canvas.paste(crop, (40, 112))
    d = ImageDraw.Draw(canvas)

    legend = [
        ("后极部卵圆(橙虚线): 鼻侧 1PD 外扩线", (250, 160, 60)),
        ("未扣除底噪(青虚线): 高斯尾部在鼻侧泄漏 ≈20% 深度", C_OFF),
        ("扣除后(紫粗线): 软件实际输出, 鼻极贴回 1PD 线", C_ON),
    ]
    ly = 130
    for txt, col in legend:
        d.line([(650, ly), (710, ly)], fill=col, width=6)
        d.text((725, ly - 14), txt, fill=(235, 235, 240), font=F(25))
        ly += 56
    d.text((650, ly + 10), f"此样例: 鼻侧被侵蚀 {erosion_px:.0f} px ≈ {erosion_rd:.1f} 视盘半径",
           fill=(255, 230, 80), font=F(26))
    d.text((650, ly + 58), "扣除量 = 两凹口尾部在鼻极的残留深度(nasal_depth_floor)",
           fill=(190, 190, 200), font=F(24))

    d.text((40, 15), "鼻侧底噪扣除(nasal_depth_floor)对比: 不扣除时, 上下凹口的高斯尾部把鼻侧 1PD 边缘向内侵蚀",
           fill=(240, 240, 240), font=F(29))
    d.text((40, 58), "扣除后: 鼻极方向 depth→0, 轮廓贴回后极部卵圆 —— 鼻侧 1PD 余量完整保留",
           fill=(200, 200, 205), font=F(25))

    ox, oy, sx = 110, 790, W - 190
    sy = 0.45
    d.text((ox + 8, oy - 38), "r(φ) 径向剖面: 只有鼻极附近两条轮廓分离, 其余方向重合",
           fill=(210, 210, 215), font=F(25))
    for gy in range(0, 701, 100):
        y = oy + 380 - gy * sy
        d.line([(ox, y), (W - 60, y)], fill=(70, 70, 78))
        d.text((30, y - 10), f"{gy} px", fill=(150, 150, 160), font=F(22))
    deg = np.degrees(phis)

    def curve(rr, col, width=4, dash=False):
        pts = [(ox + g / 360 * sx, oy + 380 - r * sy) for g, r in zip(deg, rr)]
        if dash:
            for i in range(0, len(pts) - 6, 12):
                d.line(pts[i:i + 8], fill=col, width=width)
        else:
            d.line(pts, fill=col, width=width)

    x180 = ox + 0.5 * sx
    d.line([(x180, oy - 10), (x180, oy + 392)], fill=(120, 120, 130), width=2)
    curve(r_pp, (250, 160, 60))
    curve(r_off, C_OFF, 4, dash=True)
    curve(r_on, C_ON, 5)
    d.text((x180 - 44, oy + 398), "鼻侧 180°", fill=(180, 180, 190), font=F(24))
    d.text((ox - 30, oy + 398), "颞侧 0°", fill=(180, 180, 190), font=F(24))
    d.text((ox + sx - 76, oy + 398), "颞侧 360°", fill=(180, 180, 190), font=F(24))
    # 鼻极处侵蚀量标注
    r_hi = float(r_pp[np.argmin(np.abs(phis - math.pi))])
    r_lo = float(r_off[np.argmin(np.abs(phis - math.pi))])
    arrow(d, (x180, oy + 380 - r_hi * sy), (x180, oy + 380 - r_lo * sy),
          (255, 230, 80), 3)
    d.text((x180 - 190, oy + 380 - (r_hi + r_lo) / 2 * sy - 18),
           "尾部泄漏 ≈20%", fill=(255, 230, 80), font=F(24))
    canvas.save(os.path.join(OUT, "fig9_nasal_floor.png"))


def central_inner_points(cd, Rd, cm, Rm, p):
    from labelmyeye.dr_geometry import central_inner_polygon
    return central_inner_polygon(cd[:2], Rd, cm[:2], Rm, p)


def main() -> None:
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig3(); print("fig3 ok")
    fig4(); print("fig4 ok")
    fig5(); print("fig5 ok")
    fig6(); print("fig6 ok")
    fig9(); print("fig9 ok")


if __name__ == "__main__":
    main()
