# run_clustering.py

## 作用

对应脚本：[scripts/run_clustering.py](/media/990Pro/ML/scripts/run_clustering.py)

这个脚本负责完成课程项目中的聚类部分，使用预处理后的训练集做无监督聚类，并输出聚类评价指标。

## 输入

- `data/processed/train.csv`

## 输出

- `models/metrics/clustering_metrics.csv`

## 当前方法

- `KMeans`
- `AgglomerativeClustering`

## 当前指标

- `silhouette`
- `ARI`
- `NMI`

其中：
- `silhouette` 反映簇内部紧密度和簇间分离度
- `ARI` 和 `NMI` 用于和真实标签作参考比较，但不能把它们误写成“监督学习准确率”

## 运行方式

```bash
python scripts/run_clustering.py
```

## 记录建议

- 如果后续加入 t-SNE 可视化，请在这里补上对应图片输出路径。
- 如果修改聚类数 `n_clusters`，必须在这里说明理由。
