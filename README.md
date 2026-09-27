# 空洞填补2 ——《Local Foreground Removal Disocclusion Filling Method for View Synthesis》复现

论文：H. Liang, X. Chen, H. Xu, S. Ren, H. Cai, Y. Wang,
*Local Foreground Removal Disocclusion Filling Method for View Synthesis*,
IEEE Access, vol. 8, pp. 201286–201299, 2020, DOI 10.1109/ACCESS.2020.3036053。
**论文无开源代码**，本仓库是从零实现的完整复现，每一步都有可独立运行的脚本、断言和可视化面板。

* 详细方案、逐步规格、参数出处、歧义处理记录 → **[复现方案.md](复现方案.md)**（先读这一份）
* 参考的前向 warp / 标定实现 → 另一个复现项目 `D:\项目\空洞填补`

数据：`D:\项目\3DVideos-distrib\MSR3DVideo-Ballet`（MSR 3D Video，8 相机 × 100 帧，
1024×768，彩色 jpg + 逆深度 png + 官方 `calibParams-ballet.txt`）。

---

## 1. 方法一句话

> 空洞**不在虚拟视图上 inpaint**。先在参考图/参考深度图上把"挡住背景的那块**局部前景**"
> 挖掉，用周围背景纹理把这块挖掉的区域**预测**出来（预测结果 = 虚拟视图里该露出来的
> 遮挡层），再把它 warp 到虚拟视图去填空洞。这样 3D warping 的误差不会被采样进空洞。

论文流程图（Fig. 2）：

```
参考彩色图 ─┐
            ├─► ① 形态学深度图预处理      (III-A) 鬼影检测 + 深度修正
参考深度图 ─┘
              ├─► ② 改进的 3D warping      (III-B) 裂纹填补 + 深度测试
              ├─► ③ 空洞边缘像素分类        (III-B) Laplacian 符号 → FG/BG
              ├─► ④ 局部前景移除            (III-C) 参考图 + 参考深度图上抠掉遮挡前景
              ├─► ⑤ 被移除区域填充          (III-D) 深度预测 eq.4 + 改进 Criminisi eq.6-10
              └─► ⑥ 空洞填补与后处理        (III-E) 遮挡层 warp 回来填洞
```

## 2. 环境

只需要 Python 3 + `numpy`、`opencv-python`、`scipy`（无 matplotlib，所有可视化用 cv2 自己画）。

```powershell
$py = "C:\Users\ZHJ\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
```

> ⚠️ 工作路径含中文：**一律用 `lfrd.io_utils.imread/imwrite`**（内部 `np.fromfile` +
> `cv2.imdecode`），不要直接 `cv2.imread`。

## 3. 用法

### 3.1 一步跑完整流程 + 评价

```powershell
# 论文 BA54：参考 cam5 → 虚拟 cam4，第 0 帧
& $py run_all.py --run ba54_f000 --src_cam 5 --dst_cam 4 --frames 0

# 10 帧 + 消融（同时算"无预处理 warp"对照臂，给出鬼影消除的 dB 增益）
& $py run_all.py --run ba54 --frames 0-9 --ablation

# 换个相机对（论文 BA56：参考 5 → 虚拟 6）/ 多组一起
& $py run_all.py --run ba56 --pairs 5:6 --frames 0-9
& $py run_all.py --run multi --pairs 5:4,5:6,6:7 --frames 0-9
```

产物在 `output/<run>/`：
| 目录 | 内容 |
| --- | --- |
| `00_calib/` | 标定报告、逐像素位移场 `dx/dy` 可视化 |
| `10_preproc/` | 预处理深度图、鬼影标记、`panel_preproc.png` |
| `20_warp/` | 虚拟视图、空洞掩码、三类空洞分色、`panel_warp.png`、`panel_holetype.png` |
| `30_class/` | 空洞边缘 FG/BG 分类、背景侧、`panel_class.png` |
| `40_removal/` | 移除掩码、挖掉后的参考图、`depth_pred.png`、`panel_removal.png` |
| `50_fill/` | 预测出的遮挡层、`panel_inpaint.png` |
| `60_final/` | 最终虚拟视图、`panel_final.png`（GT 并排） |
| `panels/` | `master_cam*-f###.png` 总览对比图、放大图 |
| `eval/` | `report.md`、`metrics.json`、`progress.png`、`summary.md`（消融汇总）|
| `cam<src>-cam<dst>-f###/` | 该帧的 `10_preproc`/`20_warp`/`60_final` 副本（多帧运行时必需，避免串帧）|
| `ablation/` | 消融各臂的图与报告（`ablation.py`）|

> ⚠️ **多帧运行时，各阶段目录里只有最后一帧的产物**。需要逐帧分析/评分时：
> ```powershell
> & $py run_all.py --run ba54_seq --eval_only          # 对已有帧目录逐帧算指标
> & $py ablation.py  --run ba54_seq --frame f000       # 单帧消融（自动用帧目录的产物）
> & $py summarize.py --run ba54_seq --pair 5:4         # 汇总所有帧的消融
> & $py make_final_panel.py --run ba54_seq --frame f000  # 一张总览对比图
> & $py check_all.py --run ba54_seq                    # 全部测试 + 落盘断言
> ```

### 3.2 逐步跑（每一步都能单独执行 + 断言 + 出图）

```powershell
& $py step1_preprocess.py --run ba54_f000        # ① 形态学预处理
& $py step2_warp.py       --run ba54_f000        # ② 前向 3D warping + 空洞分类
& $py step3_classify.py   --run ba54_f000        # ③ 空洞边缘 FG/BG 分类
& $py step4_removal.py    --run ba54_f000        # ④ 局部前景移除 + ⑤.1 深度预测
& $py step5_inpaint.py    --run ba54_f000        # ⑤.2 改进 Criminisi 纹理预测
& $py step6_render.py     --run ba54_f000        # ⑥ 空洞填补 + 后处理
```

每个脚本都会打印 `[CHECK][PASS|FAIL] ...` 清单，**有 FAIL 就非零退出**，
同时把清单写到对应目录的 `checks.txt`。

### 3.3 测试

```powershell
& $py tests\test_step1_2.py      # Stage A：预处理 + warp（含 Z-buffer 暴力对照）
& $py tests\test_step3_4.py      # Stage B：分类 + 移除 + 深度预测
& $py tests\test_step5_6.py      # Stage C：inpaint + 渲染
& $py tools\make_fixtures.py --run fixture_synth   # 合成场景（768×1024，走真实 warp 路径）
```

## 4. 参数

全部参数集中在 `lfrd/config.py:RunConfig`，可用 `--config run.json` 覆盖，
每次运行会在 `output/<run>/config.json` 留档。论文值优先：

| 参数 | 默认 | 出处 |
| --- | --- | --- |
| `th` 鬼影边缘阈值 | 20 | 论文 IV-A |
| `ghost_rounds` 预处理遍数 | 2 | 论文 III-A（鬼影 1–2 px 宽） |
| `splat` / `rule` | `sub` / `zbuf` | 亚像素 + 深度测试（遮挡关系正确） |
| `crack_max_width` | 2 | 论文 II（裂纹 1–2 px） |
| `patch_size` | 9 | 论文 IV-A |
| `sizes` / `beta` / `size_rule` | `null` / `150` / `best` | 自适应 patch 尺寸链（方案 A），**实验性、默认关闭**：在 cam6→cam7 上有效（+0.46~+0.62 dB），但在 BA54 上**变差**（单 9×9 的 19.57 → 19.21 / 18.95 dB），因为 3×3 会抢先。详见 [提升空间分析.md](提升空间分析.md) §4 |
| `struct_pen` | `0` | 跨行结构惩罚（参考项目的技巧）。**实测所有取值都更差**（0 → 22.94，2 → 22.82，8 → 21.98 dB），默认关闭；保留为开关 |
| `search_w × search_h` | 160 × 120 | 论文 IV-A |
| `depth_tol`（DD 上限） | 0.2 | 式(9) |
| `alpha`（数据项归一化） | 255 | Criminisi |
| `dilate_rm`（移除掩码膨胀） | 2 | 论文 III-C |
| `fill_oofa` | False | 论文 Fig.12/13 里 OOFA 仍为黑 |

### 关键参数怎么调

| 想解决的问题 | 调什么 |
| --- | --- |
| 虚拟视图里空洞太多 | 换更小的基线（如 `--pairs 5:4` → `4:3`）；或检查深度图是否正确 |
| 前景边缘有"鬼影"（前景纹理糊到背景上） | 增大 `th` 会让预处理标记更多像素；改 `ghost_rounds` 可加宽修正带 |
| 填出来的空洞有前景纹理泄漏 | 检查 `40_removal/panel_removal.png`：移除掩码是否只含前景；再查 `50_fill/panel_inpaint.png` |
| 填出来的空洞纹理重复/糊 | `search_w/search_h` 调大（论文提到窗口太小会传播重复内容）；`patch_size` 按"比最粗结构大"来定 |
| 想连 OOFA 一起填 | `--fill_oofa`（论文 Fig.12/13 里 OOFA 是黑的，所以默认关闭；但**开启后整帧 PSNR +5.8 dB**，且不影响方法负责的区域。要和别的方法比整帧指标时**务必打开**） |

## 5. 已知限制

1. **虚拟相机必须是一个真实存在的相机位置**。本复现按论文用数据集自带的相邻相机做虚拟
   视图（这样才能与真实拍摄的 GT 算 PSNR/SSIM）。论文的场景是 2D→3D / FVV，
   虚拟相机位置由用户给定；若需要**任意中间视角**，要自行实现相机内插
   （对 `t` 线性插值、对 `R` 做 SLERP），标定文件已经提供了全部内外参。
2. **`lfrd/calib.py` 的像素坐标与图像尺寸写死为 768×1024**（与数据集一致）。
   合成用例请构造同尺寸场景，否则走不通真实 warp 路径。
3. **OOFA 默认不填**（与论文图示一致）。论文 IV-E 的整体 PSNR 是在整帧上算的，
   所以包含黑色 OOFA；我们的 `eval/report.md` 同时给出**整帧**和**仅空洞区域**两个口径。
4. FSIMc / VSI 已实现（论文 IV-E 用到），但判据仍以 PSNR/SSIM + 目视为准。

## 6. 目录

```
复现方案.md              方案 / 逐步规格 / 参数出处 / 歧义处理（主文档）
README.md                本文件
lfrd/                    算法包
  config.py                RunConfig（全部参数）
  io_utils.py              非 ASCII 安全读写、数据定位、产物目录契约
  calib.py                 标定解析、官方逆深度公式、投影/反投影、位移场
  warp.py                  前向 warp（亚像素 + Z-buffer）、backward map、裂纹、OOFA 判定
  preprocess.py            ① 形态学预处理（式 1–2）
  disocclusion.py          ② 空洞连通域统计与三类（裂纹/遮挡/OOFA）划分
  fill.py                  ③ 边缘分类（式 3）+ ④ 局部前景移除 + ⑤.1 深度预测（式 4–5）
  inpaint.py               ⑤.2 改进 Criminisi（式 6–10）
  render.py                ⑥ 空洞填补 + 后处理
  viz.py / metrics.py      可视化面板 / PSNR·SSIM·FSIMc·VSI
  cli.py / reporter.py     统一命令行与 [CHECK] 断言输出
step1..step6_*.py        各步 CLI
run_all.py               端到端 + 逐帧评价 + 消融入口
ablation.py              消融/对比臂（纯 warp、+鬼影、虚拟视图直接 inpaint、本文方法）
summarize.py             多帧消融汇总 -> eval/summary.md
make_final_panel.py      一张总览对比图（GT/纯 warp/对比方法/本文方法/差值/放大）
check_all.py             跑全部测试 + 各阶段落盘断言，输出统一汇总
tests/                   各阶段断言（自建专用 run 目录，避免跨帧串扰）
tools/                   探索性探针与合成数据
output/<run>/            产物
```

## 7. 实测结果（MSR Ballet，论文的 BA54 场景 = 参考 cam5 → 虚拟 cam4，10 帧）

### 7.1 主结果（与真实拍摄的 cam4 比较）

| 口径 | 纯 warp | 本文方法（论文原样） | **+ 时序背景** | **+ 时序 + `--fill_oofa`** |
| --- | --- | --- | --- | --- |
| 整帧 PSNR / SSIM | 16.759 / 0.7416 | 19.876 / 0.7823 | 19.939 / 0.7877 | **29.471 / 0.8298** |
| 空洞区域 PSNR / SSIM（disocclusion + 裂纹） | 7.357 / 0.0201 | 21.181 / 0.5416 | **22.858 / 0.6183** | **22.859 / 0.6183** |
| 仅 disocclusion 区域 PSNR | 7.264 | 21.099 | **22.783** | 22.783 |
| 原本有效、未改写的像素（回归检查） | 31.882 | 31.882 | 31.882 | 31.882（逐像素完全一致） |

> **时序背景建模**（论文 IV/V future work，`temporal_frames=N`）把遮挡层从"猜"变成"取真实内容"：
> 洞区 **+1.68 dB**、SSIM **+0.077**；约 30% 的移除像素在序列里确实露出过背景。
> `--fill_oofa` 再贡献整帧约 +9.5 dB（不影响方法负责的区域）。
> 组合：整帧 **16.76 → 29.47 dB（+12.71）**，洞区 **7.36 → 22.86 dB（+15.50）**，
> 而原本有效的像素**逐像素完全未改**。

```powershell
# 最终最佳配置
& $py run_all.py --run my_run --pairs 5:4 --frames 0-9 --fill_oofa `
      --config output\_temporal\temp_on.json     # temporal_frames = 100
```

空洞率 11.44 %（89946 px）= 裂纹 972 + disocclusion 47179 + OOFA 41738。
单帧端到端耗时约 32 s（1024×768，单线程）；其中 inpainting 约 2 s，时序背景模型约 8 s。

> ⚠️ **"整帧 PSNR"这一行不能拿去和别的实现直接比**：这里把**未填的黑色 OOFA**
> 也算进整帧（与论文 Fig.12/13 一致），而参考项目 `D:\项目\空洞填补` 会**把 OOFA 也填掉**
> 且用**灰度 luma**。口径不同，差值可达 6~11 dB。
> **统一口径后的对比、以及"把我们的 warp 喂给它的填充器"的正面对决结论见
> [对比分析_与参考项目.md](对比分析_与参考项目.md)**：
> 在真正需要合成的像素上，本项目 **11.21 dB** vs 参考项目 **11.39 dB（差 0.18 dB，实质平手）**；
> 在最难场景 cam3→cam0 上两者打平（20.10 vs 20.08 dB）。

### 7.2 与对比方法/消融（10 帧均值，`output/ba54_seq/eval/summary.md`）

| 配置 | 整帧 PSNR | 空洞区域 PSNR | 空洞区域 SSIM |
| --- | --- | --- | --- |
| 纯 warp（原始深度，无预处理） | 16.791 | 19.758 | 0.9287 |
| 纯 warp（+ 形态学预处理，III-A） | 16.759 | 19.484 | 0.9253 |
| 直接在虚拟视图上 exemplar inpainting（对比方法 [17]） | 19.167 | 26.423 | 0.9521 |
| **本文方法（完整流程）** | **19.899** | **33.916** | **0.9704** |

逐帧洞区 PSNR：纯 warp 19.0–19.7 → 对比方法 25.4–27.1 → **本文方法 33.1–35.7 dB**，
10/10 帧优于对比方法，领先 **+6.28 ~ +9.10 dB**（均值 **+7.49 dB**）。

**结论：论文的核心主张被复现并验证** —— 把遮挡前景从参考图挖掉、在参考图上预测遮挡层、
再 warp 回来填洞，明显优于直接在虚拟视图上 inpaint。差距最大的 f000：
纯 warp 19.62 → 对比方法 26.39 → 本文方法 **35.49 dB**。

**但形态学鬼影预处理（论文 III-A）在本数据上无可测收益**：在"两臂都 warp 出来的公共
有效像素"上 32.788 → 32.772 dB（**−0.016 dB**），洞区 −0.27 dB。原因是这组结构化光深度
本身很干净（实测轮廓处"背景值混入前景"仅 1.4 万像素、约 1.8 %），修复的机会窗口很小。
论文自己也只是说鬼影会干扰后续填洞，未给出该项的单独增益。这一点如实记录，不做美化。

### 7.3 可核对的关键量化证据

| 证据 | 数值 |
| --- | --- |
| 预处理标记像素（式 1） | 14513 px（f000），被标记像素**只增不减**，`P==0` 的 158 px 全部保持不变 |
| 鬼影带 Laplacian 能量 | 303.18 → 133.57（−56 %） |
| Z-buffer 正确性 | 96×96 裁剪与暴力实现逐像素一致（覆盖 9216/9216，胜者 3252/3252） |
| 空洞边缘分类（式 3） | 8 个 disocclusion、3815 边界像素；Laplacian 规则与深度交叉校验一致率 **0.952**；FG 边缘均值 P=171.4 > BG 86.7 |
| 局部前景移除（III-C） | 50390 px，全部落在该 disocclusion 自己的前景层内；逐 run 宽 38–85 px（不是整个前景物体） |
| 深度预测（式 4–5） | 1380 个 run；`nearest` 兜底 **0** 次；预测中位数 93.0 vs 可见背景 92.0；0 个分量把前景深度留在预测层里 |
| 遮挡层重投影覆盖 | 面积加权 **0.916**（中位 0.964）的 disocclusion 由"预测遮挡层"填上 |
| 改进 Criminini（式 6–10） | 1721 次迭代、1.12 ms/次；**从前景层取样的比例 0.74 %**（关掉深度约束升到 9.1 %，关掉背景项图像差分 80.9 %） |
| 最终合成 | disocclusion 46519 px 全部填满；有效像素被改写 0 px |

### 7.4 已知限制（诚实记录）

1. **BA56（参考 cam5 → 虚拟 cam6，大基线、内容主要向左移动）尚未完全打通**：分类阶段
   正确地把背景侧判为 −1（14 个分量全部），但移除带的个别小分量跨了两个深度层、
   参考足迹 IoU 偏低（最小 0.019），step4 的内部一致性断言会失败。论文自己也说 BA56
   指标低于 BA54，因为 view 6 里露出的纹理在 view 5 中根本不存在。**主结果用 BA54**，
   BA56 需要把"移除带只含同一深度层"做成硬约束后才能稳定。
2. **OOFA 默认不填**（与论文图示一致）。整帧 PSNR 因此被 41738 个黑像素拉低；
   需要整帧好看时用 `--fill_oofa`（f000 整帧 19.96 → 29.62 dB，但这已超出论文范围）。
3. **鬼影预处理在本数据上无收益**，见 7.2。
4. **虚拟相机必须是真实存在的相机位置**（本复现用数据集相邻相机作为虚拟视图，才能与
   真实拍摄算 PSNR/SSIM）。任意中间视角需要自行实现相机内插（标定已给全内外参）。
5. `lfrd/calib.py` 的像素坐标与图像尺寸写死 768×1024（与数据集一致），合成用例需同尺寸。

### 7.5 扩展验证：5 组相机对（各前 5 帧，130 项阶段断言均通过）

为确认方法不是只在 BA54 上凑巧有效，补测了 4 个新视角（含反向视角、更大基线、论文的 33% 空洞场景）：

| 相机对 | 基线 | 空洞率 | 洞区 disocc | 整帧 warp→本文 | **洞区 warp→本文** | 洞区增益 | 对比方法 [17] | **本文 − 对比** |
|---|---|---|---|---|---|---|---|---|
| cam5→cam4（BA54，论文同款） | 3.9 | 11.4% | 46.6k px | 16.83 → **19.93** | 7.42 → **22.20** | **+14.8 dB** | 14.51 | **+7.7 dB** |
| cam4→cam5（反向视角） | 3.9 | 10.0% | 50.5k px | 17.06 → **21.39** | 7.05 → **20.93** | **+13.9 dB** | 13.58 | **+7.4 dB** |
| cam5→cam6（BA56，大基线） | 3.9 | 8.4% | 49.6k px | 17.41 → **22.79** | 6.80 → **20.54** | **+13.7 dB** | 13.20 | **+7.3 dB** |
| cam6→cam7（大基线） | 3.3 | 8.0% | 45.9k px | 17.81 → **22.66** | 7.17 → **23.11** | **+15.9 dB** | 13.96 | **+9.2 dB** |
| cam3→cam0（论文 33% 空洞的极端场景） | 10.9 | **27.3%** | 104.5k px | 12.53 → **15.28** | 7.01 → **20.64** | **+13.6 dB** | 14.90 | **+5.7 dB** |

* 5 组相机对、25 帧，**洞区 PSNR 全部大幅提升（+13.6 ~ +15.9 dB）**，且 **25/25 帧都优于"直接在虚拟视图 inpaint"的对比方法**（+5.7 ~ +9.2 dB）。
* 反向视角 cam4→cam5 同样有效（空洞开在前景左侧、背景侧为 −1），说明实现没有写死 BA54 的方向。
* 极端场景 cam3→cam0：空洞占 27.3%（10.4 万像素），仍然填满并提升 13.6 dB；整帧提升 +2.75 dB。
  该场景下式(3) 的 Laplacian 规则与深度交叉校验一致率降到 0.72（基线大、深度图噪声大，
  二阶导不稳），已在报告中如实记录 —— 这是数据属性，不是实现缺陷。
* 所有帧的"原本有效像素"逐像素完全不变（回归检查）。

**总览图**：`overview.png`（5 组相机对同一帧的 GT / 填充区域 / 纯 warp / 本文方法 / 差值，一图看完）。

### 7.6 单组明细与逐帧曲线

* `output/ba54_seq/panels/master_cam5-cam4-f000.png` —— 单组最详细的对比（含 4 个放大对照）
* `output/_compare/summary.md` —— 5 组 × 5 帧的完整指标表
* `output/_compare/q_*_cam*-f000.png` —— 每组一张对比图
* `output/<run>/eval/report.md`、`eval/summary.md`、`eval/progress.png`

### 7.7 逐步可视化（给人工核对用）

* `output/ba54_seq/panels/master_cam5-cam4-f000.png` —— **一图看全**：
  GT / 纯 warp / 对比方法 / 本文方法 / 两张差值图 / 填充区域 / 4 个放大对照。
  放大对比里可以直接看到：对比方法在窗帘处留下竖直条带伪影，本文方法把背景纹理接续过去。
* `output/ba54_seq/10_preproc/panel_preproc.png` + `panels/panel_preproc_zoom.png`（鬼影修正）
* `output/ba54_seq/20_warp/panel_warp.png`、`panel_holetype.png`（三类空洞分色）
* `output/ba54_seq/30_class/panel_class.png`（红=FG 边缘、绿=BG 边缘）
* `output/ba54_seq/40_removal/panel_removal.png`、`panel_depth_pred.png`
* `output/ba54_seq/50_fill/panel_inpaint.png`
* `output/ba54_seq/60_final/panel_final.png`
* `output/ba54_seq/eval/report.md`（逐帧表）、`eval/summary.md`（消融汇总）、`eval/progress.png`

