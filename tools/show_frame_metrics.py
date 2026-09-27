"""Show the measured metrics of one per-frame directory."""
import json
import os
import sys

sys.path.insert(0, r"D:\项目\空洞填补2")
sys.stdout.reconfigure(encoding="utf-8")

d = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\项目\空洞填补2\output\ba54_temporal__cam5-4-f008"
print("== " + os.path.basename(d))

mp = os.path.join(d, "60_final", "metrics.json")
if os.path.isfile(mp):
    with open(mp, encoding="utf-8") as fh:
        m = json.load(fh)
    print("\n区域指标 (RGB PSNR 用各自区域像素平均):")
    for k, v in m.items():
        if isinstance(v, dict):
            print("  %-8s n=%7s   ours %7.3f dB / SSIM %.4f    plain warp %7.3f dB"
                  "   gain %+7.3f"
                  % (k, v.get("n_px"), v.get("ours_psnr", float("nan")),
                     v.get("ours_ssim", float("nan")), v.get("warp_psnr", float("nan")),
                     v.get("psnr_gain", float("nan"))))

fm = os.path.join(d, "60_final", "final_meta.json")
if os.path.isfile(fm):
    with open(fm, encoding="utf-8") as fh:
        f = json.load(fh)
    print("\n填洞统计:")
    print("  hole_all %d = 裂纹 + disocclusion %d + OOFA %d"
          % (f.get("hole_all_px", 0), f.get("hole_disocc_px", 0), f.get("hole_oofa_px", 0)))
    print("  从遮挡层填入 %d px；后处理兜底 %d px；剩余 %d px；OOFA 保留 %d"
          % (f.get("filled_from_occlusion_px", 0), f.get("postprocess_filled_px", 0),
             f.get("leftover_px", 0), f.get("oofa_px", 0)))

ip = os.path.join(d, "50_fill", "inpaint_meta.json")
if os.path.isfile(ip):
    with open(ip, encoding="utf-8") as fh:
        i = json.load(fh)
    print("\n遮挡层预测 (step5):")
    for k in ("removed_px", "n_iters", "seconds", "depth_pred_median_inside_removed",
              "visible_background_level", "exemplar_source_depth_median",
              "exemplar_in_foreground_fraction", "depth_layer_used"):
        if k in i:
            v = i[k]
            print("  %-36s %s" % (k, round(v, 4) if isinstance(v, float) else v))

rp = os.path.join(d, "40_removal", "removal_stats.json")
if os.path.isfile(rp):
    with open(rp, encoding="utf-8") as fh:
        r = json.load(fh)
    print("\n局部前景移除 (step4):")
    print("  removed %s px, case_hist %s, pred case_hist %s"
          % (r.get("removed_px"), (r.get("removal_stats") or {}).get("case_hist"),
             (r.get("pred_stats") or {}).get("case_hist")))
