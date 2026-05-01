# prepare_data.py

## 作用

对应脚本：[scripts/prepare_data.py](/media/990Pro/ML/scripts/prepare_data.py)

这个脚本负责把原始 `data/raw/data.csv` 读取、清洗列名、规范化标签，并生成后续实验统一使用的训练集和测试集。

它是整个项目的第一步，其他脚本默认依赖它的输出。

## 输入

- `data/raw/data.csv`

## 输出

- `data/interim/cleaned.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`

## 当前行为

- 使用 `;` 读取 CSV
- 清理列名里的 BOM、制表符和回车
- 清理 `Target` 中的尾部空白
- 使用分层抽样拆分训练集和测试集

## 运行方式

```bash
python scripts/prepare_data.py
```

## 记录建议

- 每次如果修改了拆分比例、随机种子或标签清洗逻辑，都要在这里补一条变更说明。
- 如果后续加入特征筛选或额外清洗，也应先写在这里，再改代码。
