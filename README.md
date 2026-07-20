# LabelMyEye

面向术前/术后眼底图病灶配准的标注工具。兼容 LabelMe 多边形 JSON；支持单图标注，也支持 **Before + After** 同时加载并对齐后导出相对于 Before 的标注。

## 运行

```bash
pip install -r requirements.txt
python run.py
```

或：

```bash
python -m labelmyeye
```

## 功能

| 功能 | 说明 |
|------|------|
| 多边形标注 | 单击加点，Enter / 右键 / 双击闭合 |
| 多边形编辑 | 拖顶点、拖整体、点边插点、删顶点、改标签（F2 / 右键菜单） |
| LabelMe 读写 | 加载/保存标准 LabelMe `polygon` JSON（不写入巨大的 `imageData`） |
| 单图模式 | File → Open Image / Open LabelMe JSON（与 LabelMe 类似） |
| 双图配准 | File → Open Dual Project，或 Quick Dual from Folder |
| 自由缩放 | 滚轮缩放视图；Shift+拖拽 / 中键平移 |
| Before 配准 | 模式选「Register Before」：拖拽平移；Ctrl+滚轮缩放；Alt+滚轮旋转 |
| Undo / Redo | Ctrl+Z / Ctrl+Y |
| 导出 | **Export Before-relative JSON**：双图模式下把 After 层多边形按当前配准变换到 Before 坐标系后导出 |

## 推荐工作流（双图）

1. `File → Quick Dual from Folder…` 选择含 before/after 图片与 JSON 的文件夹。
2. 调 Before 透明度，进入 **Register Before**，把解剖标志对齐。
3. 需要时在 After 或 Before 层继续画多边形。
4. `Ctrl+E` 导出相对于 Before 的 LabelMe JSON。
5. 可选：`Save Dual Project` 保存配准变换与图层信息（`*.lmye.json`）。

## 快捷键

- `Wheel` 视图缩放  
- `Shift+Drag` / 中键 平移  
- `Ctrl+Wheel` 缩放 Before 图（配准）  
- `Alt+Wheel` 旋转 Before 图  
- `Enter` / 右键 闭合多边形  
- `Esc` 取消当前多边形 · `Del` 删除选中  
- `F` 适应窗口  
- `Ctrl+Z` / `Ctrl+Y` 撤销 / 重做  
- `Ctrl+S` 保存 · `Ctrl+E` 导出 Before 坐标系 · `Ctrl+D` 打开双图  

## 安装包（Windows）

本地构建后位于：

`dist\installer\LabelMyEye-Setup-0.1.0.exe`

双击安装即可，无需单独安装 Python。重新打包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

## 隐私说明

**请勿将真实患者影像、标注 JSON、病历摘要或含姓名/病历号的元数据提交到本仓库。** 此类文件已在 `.gitignore` 中排除。仅使用脱敏样例或本地私有数据。
