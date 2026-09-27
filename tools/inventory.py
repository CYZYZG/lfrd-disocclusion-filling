"""Inventory the project and classify every file as needed / redundant.

Read-only: it prints the classification and the reason, so the tidy-up can be reviewed before
anything is moved.

    python tools/inventory.py [--csv out.csv]
"""
import argparse
import ast
import csv
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {".git", "output", "__pycache__", ".venv"}

# Files the PACKAGE or the TESTS actually need.
CORE_LIB = {
    "lfrd/__init__.py", "lfrd/api.py", "lfrd/calib.py", "lfrd/cli.py", "lfrd/config.py",
    "lfrd/disocclusion.py", "lfrd/fill.py", "lfrd/inpaint.py", "lfrd/io_utils.py",
    "lfrd/metrics.py", "lfrd/preprocess.py", "lfrd/render.py", "lfrd/reporter.py",
    "lfrd/viz.py", "lfrd/warp.py", "lfrd/temporal.py", "lfrd/refguide.py",
}
CORE_STAGES = {f"step{i}_{n}.py" for i, n in
               ((1, "preprocess"), (2, "warp"), (3, "classify"), (4, "removal"),
                (5, "inpaint"), (6, "render"))} | {"run_all.py", "check_all.py",
                                                   "compare_pairs.py", "ablation.py",
                                                   "summarize.py", "overview_all.py",
                                                   "make_final_panel.py"}
CORE_TESTS = {"tests/test_step1_2.py", "tests/test_step3_4.py", "tests/test_step5_6.py",
              "tests/__init__.py"}
CORE_TOOLS = {
    # referenced by README / 提升空间分析.md / 接口使用说明.md recipes, or by each other
    "tools/oracle_ceiling.py", "tools/error_anatomy.py", "tools/second_view.py",
    "tools/shift_probe.py", "tools/depth_check.py", "tools/metric_parity.py",
    "tools/head_to_head.py", "tools/h2h_fair.py", "tools/h2h_panel.py",
    "tools/region_breakdown.py", "tools/ab_cascade.py", "tools/ab_template_fg.py",
    "tools/two_source_probe.py", "tools/verify_readme_numbers.py",
    "tools/depth_refine_probe.py", "tools/refguide_probe.py", "tools/refguide_sweep.py",
    "tools/refguide_selective.py", "tools/temporal_bg.py", "tools/temporal_bg2.py",
    "tools/temporal_layer.py", "tools/temporal_agg.py", "tools/temporal_extend.py",
    "tools/temporal_colour_check.py", "tools/photometric_check.py", "tools/photo_correct.py",
    "tools/photo_correct_probe.py", "tools/photo_spatial.py", "tools/photo_residual.py",
    "tools/photo_strength_check.py", "tools/stage6_accounting.py", "tools/coverage_breakdown.py",
    "tools/gap_decompose.py", "tools/texture_energy.py", "tools/texture_h2h.py",
    "tools/sibling_repair_check.py", "tools/crack_detect_compare.py", "tools/crack_fill_probe.py",
    "tools/crack_claim_ceiling.py", "tools/route_compare.py", "tools/sibling_input_ab.py",
    "tools/warp_valid_quality.py", "tools/ref_availability.py", "tools/ref_reselect.py",
    "tools/ref_depth_consistent.py", "tools/selector_search.py", "tools/source_select_oracle.py",
    "tools/shift_measure.py", "tools/shift_on_defects.py", "tools/over_texture_check.py",
    "tools/depth_clean_check.py", "tools/depth_position_check.py", "tools/fix_position.py",
    "tools/make_gallery.py", "tools/artifact_map.py", "tools/make_index.py",
    "tools/final_compare.py", "tools/show_configs.py", "tools/show_frame_metrics.py",
    "tools/verify_no_dl.py", "tools/panel_final_compare.py", "tools/make_fixtures.py",
    "tools/smoke_core.py", "tools/probe_cascade.py", "tools/probe_ghost.py",
}
CORE_DOCS = {"README.md", "复现方案.md", "对比分析_与参考项目.md", "提升空间分析.md",
             "纹理区分析与参考项目优势溯源.md", "传统方法边界.md", "结果查看指南.md",
             "接口使用说明.md", ".gitignore", "examples_fill_holes.py"}
# Experimental / one-off / superseded; kept but out of the way.
ARCHIVE_HINT = {
    "lfrd/learned.py": "字典/稀疏表示路线，实测与时序冲突，未接入流程",
    "tools/learned_probe.py": "同上，该路线的探针",
    "tools/learned_sweep.py": "同上",
    "tools/probe_learned_env.py": "一次性环境侦察（结论已写进文档）",
    "tools/probe_models_deep.py": "一次性全盘模型搜索",
    "tools/sweep_planA.py": "被 ab_cascade.py 取代的参数扫描",
    "tools/score_planA.py": "早期方案A打分脚本，已被 final_compare 取代",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    # which module/script names are mentioned anywhere (crude but effective)
    text = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for f in filenames:
            if f.endswith((".py", ".md")):
                try:
                    text.append(open(os.path.join(dirpath, f), encoding="utf-8").read())
                except Exception:                                    # noqa: BLE001
                    pass
    blob = "\n".join(text)

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for f in sorted(filenames):
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            sz = os.path.getsize(p)
            if rel in CORE_LIB | CORE_STAGES | CORE_TESTS | CORE_DOCS:
                cat, why = "KEEP", "核心：库/阶段/测试/主文档"
            elif rel in CORE_TOOLS:
                cat, why = "KEEP", "诊断工具：被文档或其它工具引用"
            elif rel in ARCHIVE_HINT:
                cat, why = "ARCHIVE", ARCHIVE_HINT[rel]
            elif rel.startswith("paper_figs/") or rel.endswith(".pdf"):
                cat, why = "KEEP(ignored)", "论文材料，已被 .gitignore 排除"
            elif rel.startswith("tests/"):
                cat, why = "KEEP", "测试"
            elif rel.startswith("tools/"):
                cat, why = "KEEP", "工具（保留）"
            elif rel.endswith(".pyc"):
                cat, why = "ARCHIVE", "编译缓存"
            else:
                n = os.path.basename(rel)
                mentioned = n in blob and blob.count(n) > 1
                cat = "KEEP" if mentioned else "REVIEW"
                why = "被其它文件引用" if mentioned else "未在其它文件中出现，请人工确认"
            rows.append((cat, rel, sz, why))

    order = {"KEEP": 0, "KEEP(ignored)": 1, "ARCHIVE": 2, "REVIEW": 3}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1]))
    tot = 0
    for cat, rel, sz, why in rows:
        if cat != "KEEP(ignored)":
            tot += sz
        print(f"{cat:14s} {rel:56s} {sz:9,d} B  {why}")
    print()
    counts = {}
    for cat, rel, sz, why in rows:
        counts[cat] = counts.get(cat, 0) + 1
    print("分类统计:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"跟踪文件总大小: {tot / 1024:.0f} KB")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["category", "path", "bytes", "reason"])
            w.writerows(rows)
        print("written:", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
