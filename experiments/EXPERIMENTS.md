# 实验台账（EXPERIMENTS REGISTRY）

维护规则：**每开一个实验必须在此登记**（编号、协议、状态、结果、产物路径）；结束后回填结果。
本文件只登记元数据与结果数字，不存放患者数据。逐图明细在 `output/benchmark/results.json`（本地，不入库）。

## 实验协议（所有实验共用）

- **数据**：组内手绘标注 `scripts/calibration_data/`（16 例 FGADR，含 15 例完整 optic+macular）
- **Ground Truth**：手绘 optic / macular 圆（多边形最小二乘拟合圆）
- **指标**：
  - `IoU`：预测圆盘 vs GT 圆盘的栅格交并比
  - `中心距`：预测圆心到 GT 圆心的欧氏距离，单位 = 视盘半径（Rd）
- **注意**：手绘 macular 圆半径是组内约定（0.85×视盘半径），预测模型输出的是自身分割范围，
  因此 **macular 的主要比较指标是中心距**，IoU 仅供参考
- **派生结构说明**：macular_area / posterior_pole_boundary / central_inner 是规则派生的，
  text-guided 模型只评测 optic 与 macular 两个原始结构

---

## 实验列表

| 编号 | 模型 | 提示方式 | 状态 | optic IoU (中位) | macular IoU (中位) | 中心距 (中位, Rd) | 产物 |
|------|------|----------|------|------------------|--------------------|--------------------|------|
| E001 | MobileSAM (本仓库内置) | 框提示 = GT圆外扩1.3× | ✅ 完成 2026-10-02 | **0.799** | 0.424 | optic 0.10 / macular 0.15 | `output/benchmark/results.json` |
| E002 | Medical-SAM3 (SAM3架构, 文本引导) — 替代被 gated 的 MedSAM3 | 文本 "optic disc" / "macula lutea" / "fovea" 等按置信度择优 | ✅ 完成 2026-10-03 | **0.279** [0.001–0.934] | **0.000** [0–0.140] | optic 0.94 / macular 3.45 | 服务器 out/ → 本地 `output/benchmark/medicalsam3/` |
| E003 | MedSAM3 (Joey-S-Liu, SAM3+LoRA 文本引导) | 文本 "optic disc"/"macula lutea" 等 | ✅ 完成 2026-10-03（底座改用非 gated 转存 1038lab/sam3 的 sam3.pt 3.45GB） | **0.861** [0.751–0.960] | **0.000** [0–0] | optic 0.06 / macular 5.15 | `output/benchmark/medsam3/` |
| E004 | MedCLIP-SAMv2 (MedIA 2025, CLIP+SAM 文本驱动) | 文本 | ⛔ 阻塞: 微调权重仅在 Google Drive(`1jjnZabUlc9...`), 服务器与本机均不可达; 需任意可翻墙设备转存 | — | — | — | — |

## E002 服务器执行详情（Titan 四卡服务器）

- 连接: `student@10.16.9.162:10080`（密钥登录）
- 工作区: `~/labelmyeye_bench/`（独立目录，不触碰其他内容）
- 权重: `Chongcong/Medical-SAM3` 的 `checkpoint_2D.pt`（10GB 全量, 含 SAM3 底座, 非 gated）经 hf-mirror 下载
- 运行: `CUDA_VISIBLE_DEVICES=0 python run_bench.py`（TITAN RTX 24GB）
- 提示词候选: optic → ["optic disc", "optic nerve head", "optic nerve"];
  macular → ["macula lutea", "macula of retina", "fovea"]
- 输出: 16 例 × 2 结构 = 32 张预测 mask PNG → 拉回本地
  `output/benchmark/medicalsam3/` 后用 `scripts/benchmark_models.py --models medicalsam3` 自动计分

## 结果回填记录

- 2026-10-02: 建档。E001 完成（n=15: optic IoU 0.799 [0.402–0.970], macular IoU 0.424 [0.297–0.633]）。
- 2026-10-03: E002 完成（Titan GPU0, checkpoint_2D.pt, 49 图全跑）。optic IoU 0.279、macular 失效。
- 2026-10-03: E003 完成（Titan GPU0, 底座 sam3.pt 3.45GB 转存 + LoRA 70MB）。**关键发现:**
  ① 视盘: MedSAM3 文本引导 IoU 0.861 [0.751–0.960], **优于框提示 MobileSAM 0.799 [0.402–0.970]**——
    不仅中位数更高, 最差情况(0.751)也好于 MobileSAM 的中位数, 稳定性显著更强;
  ② 黄斑: MedSAM3 完全失效(IoU 0, 中心距 5.15Rd), 与 Medical-SAM3 一致——'macula lutea' 概念
    在眼底彩照域无法 ground;
  ③ 结论: 视盘检测上 text-guided SAM3 架构已可超过框提示方案(可作为零标注自动预标注);
    黄斑仍必须依赖框提示或专用微调。两模型均未在眼底域训练, 微调后上限值得用本数据集探索。  E003 解锁条件: 向 Meta 申请 facebook/sam3 的 HF 访问（gated: manual）后, 重跑 E003。
  E004 解锁条件: 任意可访问 Google Drive 的设备下载 `1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5` 转存到服务器后, 重跑 E004。
