# prepare_data.py

## 作用

对应脚本：[scripts/preprocessing/prepare_data.py](/media/990Pro/ML/scripts/preprocessing/prepare_data.py)

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

## Appendix 类别处理校对

课程 handout 和原论文 Appendix 都把该数据集描述为 mixed-type tabular data。当前代码按字段语义处理类别列，而不是按 pandas dtype 自动判断，因为原始 CSV 里的类别编码和连续/计数特征基本都会读成数值 dtype。

语义型拆分在 `src.preprocessing.split_feature_columns()` 中实现，类别列来源于 `src.config.CATEGORICAL_COLUMNS`。这些列包括 `Marital status`、`Application mode`、`Course`、`Previous qualification`、`Nacionality`、父母学历/职业、`Gender`、`Daytime/evening attendance`，以及 yes/no 二元标志列。

`Application order` 没有放入 categorical 列表。PDF Table 1 将它标为 `Numeric/ordinal`，所以当前 preprocessing 保持它为数值/有序特征。`Previous qualification (grade)`、`Admission grade`、年龄、课程单元数量/成绩和宏观经济变量也按 numeric 处理。

注意：Appendix 中的部分具体编码表与当前 `data/raw/data.csv` 的取值不完全一致，例如 `Application mode`、`Course`、`Previous qualification`、`Nacionality` 和父母职业/学历编码都有超出 Appendix 范围的值。因此当前项目只使用 Appendix 的字段类型语义，不直接用 Appendix 的逐值标签映射。

## 运行方式

```bash
python scripts/preprocessing/prepare_data.py
```

## 记录建议

- 每次如果修改了拆分比例、随机种子或标签清洗逻辑，都要在这里补一条变更说明。
- 如果后续加入特征筛选或额外清洗，也应先写在这里，再改代码。
