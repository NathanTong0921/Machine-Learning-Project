# Data Dictionary

## 数据集

- 名称：Predict Students' Dropout and Academic Success
- 样本数：4424
- 输入特征数：36
- 标签：`Dropout` / `Enrolled` / `Graduate`

## 特征分组建议

### 入学与人口学

- Marital status
- Application mode
- Application order
- Course
- Daytime/evening attendance
- Previous qualification
- Previous qualification (grade)
- Nacionality
- Age at enrollment
- International

### 家庭社会经济背景

- Mother's qualification
- Father's qualification
- Mother's occupation
- Father's occupation
- Displaced
- Educational special needs
- Debtor
- Tuition fees up to date
- Gender
- Scholarship holder

### 第一学期表现

- Curricular units 1st sem (credited)
- Curricular units 1st sem (enrolled)
- Curricular units 1st sem (evaluations)
- Curricular units 1st sem (approved)
- Curricular units 1st sem (grade)
- Curricular units 1st sem (without evaluations)

### 第二学期表现

- Curricular units 2nd sem (credited)
- Curricular units 2nd sem (enrolled)
- Curricular units 2nd sem (evaluations)
- Curricular units 2nd sem (approved)
- Curricular units 2nd sem (grade)
- Curricular units 2nd sem (without evaluations)

### 其他数值变量

- Admission grade
- Unemployment rate
- Inflation rate
- GDP

## 备注

- CSV 使用 `;` 分隔
- 列名包含 BOM/制表符/回车，读入后应先清洗
- 很多整数列本质上是类别编码，不应默认视为连续变量
