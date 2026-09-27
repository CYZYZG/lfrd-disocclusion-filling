# archive/ —— 归档的代码，**没有删除**

这里的文件都是**实验性、一次性或被取代**的，不属于交付范围，但保留下来以便追溯。
**当前 pipeline 与测试完全不依赖它们**（已用 `tools/inventory.py` 与引用扫描确认）。

---

## 为什么归档而不是删除

* 每一个归档文件都对应文档里的一个**实测结论**（尤其是否定结论）；
* 结论一旦有人质疑，可以立刻拿这些脚本重跑一遍复现；
* 但它们不属于"调用接口"或"核心流程"，放在 `tools/` 里会干扰阅读。

---

## 目录内容

### `learned/` —— 字典/稀疏表示路线（未接入流程）

| 文件 | 说明 |
| --- | --- |
| `lfrd_learned.py` | 逐通道 PCA 字典 + Tikhonov 最小二乘的补全实现。**原位置 `lfrd/learned.py`** |
| `learned_probe.py` | 单帧验证探针 |
| `learned_sweep.py` | 参数扫描 + 与时序叠加测试 |

**实测结论**（`传统方法边界.md`）：
不用时序时把移除区填到 **+0.44 dB**；叠在时序模型上反而 **−0.78 dB**（与时序冲突，
和 `refguide` 同一个模式）。**未接入 pipeline，`step1..6` 不 import 它。**

> 注意：这是**传统方法**（PCA/SVD 字典，非神经网络），归档的原因只是"实测无收益 + 未采用"。

### `planA/` —— 早期参数扫描（已被取代）

| 文件 | 说明 |
| --- | --- |
| `sweep_planA.py` | 方案 A 的参数网格扫描。**被 `tools/ab_cascade.py` 取代** |
| `score_planA.py` | 早期打分脚本（从日志里抠数字，易出错）。**被 `tools/final_compare.py` 取代** |

**实测结论**：自适应 patch 尺寸与跨行结构惩罚在本数据上都是负收益，默认关闭
（见 `提升空间分析.md` §4）。

### `probe/` —— 一次性环境侦察

| 文件 | 说明 |
| --- | --- |
| `probe_learned_env.py` | 探测本机是否有 torch/tensorflow/onnxruntime 与补全模型权重 |
| `probe_models_deep.py` | 全盘搜索模型文件 + 检查 pip 是否可联网 |

**实测结论**：本机**没有任何深度学习框架**，**没有任何补全模型权重**，**pip 离线**。
所以"学习型补全"在这个环境下不可行——这个结论已写入 `传统方法边界.md` §0，
脚本本身不再需要。

---

## 如果你要重新启用某个归档文件

它们依赖 `lfrd` 包，运行时需要把项目根目录放进 `sys.path`：

```python
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
sys.path.insert(0, PROJECT)

# 例如归档的字典学习实现（原 lfrd/learned.py，现在文件名不同）：
import importlib.util
spec = importlib.util.spec_from_file_location(
    "learned", os.path.join(PROJECT, "archive", "learned", "lfrd_learned.py"))
learned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learned)
```

或者直接把它拷回 `lfrd/learned.py`，再运行同目录的探针脚本。

---

## 这个分类是怎么来的

```powershell
$py = "python"
& $py tools\inventory.py --csv output\_inventory.csv
```

`tools/inventory.py` 会遍历项目，把每个文件归为
`KEEP`（核心库/阶段/测试/主文档/被引用的诊断工具）、
`KEEP(ignored)`（论文材料，已被 `.gitignore` 排除）、
`ARCHIVE`（本目录）、
`REVIEW`（需人工确认）四类，并给出理由。
