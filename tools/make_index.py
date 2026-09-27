"""Generate output/index.html: every key figure in viewing order, one page.

    python tools/make_index.py [--run ba54_best] [--frame f000]

The page is plain static HTML with relative <img> paths, so it can be opened directly from
the filesystem.  It is regenerated from whatever is on disk, so it never lists a missing file.
"""
import argparse
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils

# (relative path, title, what to look for)
SECTIONS = [
    ("最终效果（先看这些）", [
        ("{best}/panels/panel_final_compare.png", "① 论文原样 vs 时序+OOFA（含空洞放大）",
         "放大格里：论文原样把窗帘纹理糊掉，时序版把窗帘皱褶和横栏杆结构还原出来"),
        ("{best}/eval/progress.png", "② 逐帧 PSNR 曲线", "每帧的整帧/洞区指标"),
        ("../overview.png", "③ 5 组相机对总览", "GT / 填充区域 / 纯 warp / 本文方法 / 差值"),
        ("_compare/ba54_seq_cam5-cam4-f000.png", "④ BA54 单帧：论文原样",
         "五图对比 + 差值 + 填充区域"),
        ("_compare/ba54_temporal_cam5-cam4-f000.png", "⑤ BA54 单帧：+时序背景",
         "同一帧、同一 warp，只换遮挡层来源"),
        ("_compare/q_30_cam3-cam0-f000.png", "⑥ 极端场景 cam3→cam0（27% 空洞）", ""),
        ("_h2h/q_67_cam67_f000_panel.png", "⑦ 与参考项目的正面对决", "两边填同一份 warp"),
    ]),
    ("逐阶段（每一步一图看懂）", [
        ("{best}/{fd}/10_preproc/panel_preproc.png", "① 深度预处理 + 鬼影修正",
         "前景块内部的暗斑被抹平，轮廓没被整体变胖"),
        ("{best}/{fd}/20_warp/panel_warp.png", "② 3D warping", "虚拟视图与位移场"),
        ("{best}/{fd}/20_warp/panel_holetype.png", "② 空洞分类",
         "裂纹=黄、遮挡=红、OOFA=蓝"),
        ("{best}/{fd}/30_class/panel_class.png", "③ 空洞边缘 FG/BG 分类",
         "红=前景边缘、绿=背景边缘，含背景侧箭头与一致率"),
        ("{best}/{fd}/40_removal/panel_removal.png", "④ 局部前景移除",
         "只挖掉人物被空洞覆盖的那一条，不是整个人"),
        ("{best}/{fd}/40_removal/panel_depth_pred.png", "④ 深度预测",
         "预测值应落在周围背景水平，不能是人物深度"),
        ("{best}/{fd}/50_fill/panel_inpaint.png", "⑤ 遮挡层预测（改进 Criminisi）",
         "挖掉的区域被周围背景纹理接续，没从人物身上取样"),
        ("{best}/{fd}/60_final/panel_final.png", "⑥ 最终合成", "GT / 纯 warp / 结果 / 差值 / 放大"),
    ]),
]

FOLDERS = [
    ("最终结果", "ba54_best / ba54_temporal / ba54_seq / ba54_oofa",
     "最佳配置 / 只加时序 / 论文原样 / 只加 OOFA。看图请用里面的 cam5-cam4-f###/ 子目录。"),
    ("正面对决与口径分析", "_h2h / _parity / _compare",
     "与参考项目把同一份 warp 喂给两边填充器的对比、口径统一后的对比表、5 组相机对对比。"),
    ("参数实验（结论：都无收益）", "q_* / p* / AA_* / _planA / _tpl / _ab / _temporal",
     "相机对批量测试、自适应尺寸级联、模板去污染等 A/B。"),
    ("测试与合成数据", "fixture* / smoke / probe / _test_stage_*",
     "合成场景与冒烟测试的产物。"),
    ("可清理", "_verify / _dbg_f006（空目录），及上面 B/C/D 类中不需要追溯的部分",
     "约 2 GB；不影响任何结论。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="ba54_best")
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--pair", default="cam5-cam4")
    a = ap.parse_args()
    out = os.path.join(io_utils.OUTPUT_ROOT, "index.html")
    fd = f"{a.pair}-{a.frame}"

    def rel(p):
        return p.replace("{best}", a.best).replace("{fd}", fd)

    def exists(p):
        return os.path.isfile(os.path.join(io_utils.OUTPUT_ROOT, rel(p)))

    parts = ["""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>空洞填补复现 — 结果索引</title>
<style>
 body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;
      background:#14161a;color:#e6e6e6}
 header{padding:18px 26px;background:#1d2027;border-bottom:1px solid #2c313a}
 h1{margin:0 0 6px;font-size:19px}
 header div{color:#9aa4b2;font-size:13px;line-height:1.7}
 h2{margin:34px 26px 10px;font-size:16px;color:#8fd3ff;border-left:3px solid #8fd3ff;
    padding-left:10px}
 .grid{display:flex;flex-wrap:wrap;gap:16px;padding:0 26px}
 figure{margin:0;background:#1d2027;border:1px solid #2c313a;border-radius:8px;
        overflow:hidden;max-width:100%}
 figure img{display:block;width:100%;max-width:1180px;height:auto;background:#000}
 figcaption{padding:9px 12px;font-size:13px;line-height:1.6}
 figcaption b{color:#fff}
 figcaption span{color:#9aa4b2}
 code{background:#262b33;padding:1px 5px;border-radius:3px;font-size:12px}
 table{border-collapse:collapse;margin:0 26px;font-size:13px}
 th,td{border:1px solid #2c313a;padding:6px 12px;text-align:left}
 th{background:#1d2027;color:#8fd3ff}
 .note{margin:10px 26px 0;color:#9aa4b2;font-size:13px;line-height:1.8}
 .warn{color:#ffd479}
</style></head><body>
<header>
 <h1>Local Foreground Removal Disocclusion Filling — 复现结果索引</h1>
 <div>论文：Liang et al., IEEE Access 2020（无开源代码）·数据：MSR 3D Video Ballet ·
      主场景 BA54 = 参考 cam5 → 虚拟 cam4<br>
      三种配置（BA54，10 帧）：论文原样 → <b>+时序背景</b> → <b>+时序+OOFA填充</b>；
      完整说明见项目根目录 <code>结果查看指南.md</code> 与 <code>提升空间分析.md</code></div>
</header>
"""]
    parts.append("<h2>效果总览（BA54，10 帧均值）</h2>\n<table>"
                 "<tr><th>指标</th><th>纯 warp</th><th>论文原样</th><th>+时序</th>"
                 "<th>+时序+OOFA</th></tr>"
                 "<tr><td>整帧 PSNR</td><td>16.76</td><td>19.88</td><td>19.94</td>"
                 "<td><b>29.47</b></td></tr>"
                 "<tr><td>整帧 SSIM</td><td>0.742</td><td>0.782</td><td>0.788</td>"
                 "<td><b>0.830</b></td></tr>"
                 "<tr><td>洞区 PSNR</td><td>7.36</td><td>21.18</td>"
                 "<td><b>22.86</b></td><td><b>22.86</b></td></tr>"
                 "<tr><td>洞区 SSIM</td><td>0.020</td><td>0.542</td>"
                 "<td><b>0.618</b></td><td><b>0.618</b></td></tr>"
                 "<tr><td>有效像素 PSNR</td><td>31.88</td><td>31.88</td><td>31.88</td>"
                 "<td>31.88（逐像素未改）</td></tr></table>\n"
                 '<div class="note">"有效像素"一行证明填洞没有破坏原本正确的内容。</div>\n')

    for title, items in SECTIONS:
        parts.append(f"<h2>{html.escape(title)}</h2>\n<div class='grid'>\n")
        for relp, caption, note in items:
            if not exists(relp):
                continue
            p = rel(relp)
            parts.append(f"<figure><img src='{html.escape(p)}' loading='lazy'>"
                         f"<figcaption><b>{html.escape(caption)}</b>")
            if note:
                parts.append(f"<br><span>{html.escape(note)}</span>")
            parts.append(f"<br><code>{html.escape(p)}</code></figcaption></figure>\n")
        parts.append("</div>\n")

    parts.append("<h2>output 下的目录分类</h2>\n<table><tr><th>类别</th><th>目录</th>"
                 "<th>说明</th></tr>\n")
    for cat, dirs, note in FOLDERS:
        parts.append(f"<tr><td>{html.escape(cat)}</td><td><code>{html.escape(dirs)}</code>"
                     f"</td><td>{html.escape(note)}</td></tr>\n")
    parts.append("</table>\n")
    parts.append('<div class="note warn">看图请优先用带帧号的子目录 '
                 f'<code>{a.best}/{fd}/</code>：顶层 <code>20_warp/</code>、'
                 '<code>60_final/</code> 只保留最后处理过的那一帧。</div>\n')
    parts.append("</body></html>\n")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    n = sum(1 for _, items in SECTIONS for relp, _, _ in items if exists(relp))
    print(f"written: {out}   ({n} figures linked, missing ones skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
