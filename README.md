# TSP Course Experiment Framework

这是一个用于 TSP 课程实验的轻量 Python 框架。它支持从多行 txt 文件读取实例、通过统一 solver 接口运行多个算法、比较路线长度和运行时间，并预留 RL 环境接口。

## 输入格式

每行一个实例。坐标按 `x y` 成对出现，`output` 后可以放参考闭环路径：

```text
0.1 0.2 0.3 0.4 0.5 0.6 output 2 3 1 2
```

没有 `output` 时仍然可以求解和比较。

## 按规模分类数据

使用 `tool/classify_data.py` 将一个数据文件中的实例按点数拆分。默认在输入文件所在目录生成分类文件，文件名由原文件名和规模类别组成；没有实例的类别不会生成文件：

```powershell
.\.venv\bin\python.exe tool/classify_data.py data/combined_train.txt
```

上述命令会按需生成 `combined_train_small.txt`、`combined_train_medium.txt`、`combined_train_large.txt` 和 `combined_train_ultra_large.txt`。分类范围为：小规模少于 100 个点，中规模 100 至 999 个点，大规模 1000 至 10000 个点，超大规模多于 10000 个点。可以使用 `-o` 指定其他输出目录：

```powershell
.\.venv\bin\python.exe tool/classify_data.py data/combined_train.txt -o output
```

## 运行示例

当前环境使用 MSYS2 Python，虚拟环境的解释器位于 `.venv/bin/python.exe`：

```powershell
.\.venv\bin\python.exe example.py
```

算法通过 `TSPSolver` 接口实现，多个算法可以传给 `compare()` 统一运行：

```python
from tsp.compare import compare
from tsp.parser import load_instances
from tsp.solvers import NearestNeighborSolver

instance = load_instances("data/sample.txt")[0]
results = compare(instance, [NearestNeighborSolver()])
```

可在构造实例或读取数据时指定距离矩阵阈值。例如：

```python
instance = load_instances("data/sample.txt", distance_matrix_threshold=1000)[0]
```

城市数大于阈值时，`instance.distance_matrix[i][j]` 会实时计算且不缓存距离；等于或小于阈值时仍预计算。未指定阈值则保持原有的预计算行为。

绘图功能是可选的，需要安装 `requirements.txt` 中的 Matplotlib。核心解析、指标和算法对比不依赖第三方包。

## 扩展算法

在 `tsp/solvers/` 新增类并实现：

```python
def solve(self, instance: TSPInstance) -> TSPSolution:
    ...
```

RL 推理算法也可以通过同一个 `TSPSolver` 接口接入；训练环境遵循 `tsp.env.TSPEnvProtocol` 的 `reset()` / `step()` 约定。

