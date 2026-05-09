# Course_171 CTeSP Creditation Subgroup Finding

## Summary

在 CatBoost top-green subgroup 与 RF proximity embedding 的交叉检查中，当前更稳健的解释是：

> The CatBoost top-green subgroup is reproduced inside the RF proximity embedding, but the broader RF upper region also contains additional ordinary Graduate students. Therefore, the robust subgroup is the 75-sample intersection, not the entire RF upper region.

也就是说，真正值得解释的稳定子群不是 RF upper region 全部样本，而是 CatBoost 与 RF proximity embedding 共同捕捉到的 `75` 个交集样本。

## Course_171 Fact Correction

之前关于 `Course_171` 的解释需要修正。早先猜测它可能是某种内部衍生代码，例如“专业 17”的夜校管理学变体。但根据后续对葡萄牙 DGES 课程代码体系的检索，`L171` 是一个独立官方课程代码，对应：

```text
Design de Animação e Multimédia
```

即“动画与多媒体设计”本科专业。

如果这 75 个特殊样本的原始特征中 `Course_171` 占比为 100%，那么这批“前两学期课程记录近乎全 0、但最终 Graduate”的学生并不是随机异常，而很可能来自同一个课程制度背景。

## Likely Administrative Mechanism

当前解释假设是：这些样本可能是葡萄牙高等教育体系中的 CTeSP 升学通道与学分豁免机制共同造成的数据表征。

CTeSP 指：

```text
Cursos Técnicos Superiores Profissionais
```

即“高等专业技术课程”。在该体系下，学生完成 2 年制专业技术课程后，可以通过特殊录取通道升入相关本科项目。若学校承认其既有课程与学习经历，则可通过 `Creditação`，也就是学分豁免或学分等值认证，把先前学习成果折算成本科阶段基础学分。

对 `Course_171` 的解释链如下：

1. `Course_171 / L171` 对应 `Design de Animação e Multimédia`。
2. 学校体系中存在相关的 CTeSP 项目，例如多媒体与视听设计方向。
3. CTeSP 学生升入本科后，部分低年级课程可能被学分认证覆盖。
4. 因此在本科数据表中，他们第一学年可能表现为：
   - `Curricular units 1st sem (enrolled) = 0`
   - `Curricular units 1st sem (evaluations) = 0`
   - `Curricular units 1st sem (grade) = 0`
   - `Curricular units 2nd sem (...)` 同样接近 0
5. 这些 0 并不一定表示学生没有学习能力或已经 disengaged，而可能表示对应课程已被豁免，不需要在当前本科系统中重新注册、考试或产生当期成绩。

这解释了为什么这些学生在传统特征语义下看起来像“幽灵学生”，但最终标签却是 `Graduate`。

## Modeling Meaning

这个发现对模型解释很重要。

普通分类器很容易把“前两学期成绩和注册记录为 0”理解成高风险信号，因为在大多数学生中，这种模式接近 Dropout 或异常退学。但 CatBoost proximity / tree-based topology 把这 75 个样本从普通 Graduate 中剥离出来，说明树模型可能捕捉到了一个隐藏的行政子群：

```text
Course_171 + low/zero semester activity + Graduate
```

这不是简单的数学聚类巧合，而可能对应真实教务制度中的 CTeSP 升学和学分豁免路径。

因此，在报告中可以把该子群描述为：

> 基于树模型的拓扑降维不仅区分了常规学业表现强弱，还反向暴露了一个具有现实教育含义的特殊行政子群。该子群很可能由 `Course_171` 的 CTeSP 升学与学分豁免机制形成，其前两学期的 0 值不是低能力或缺课信号，而是课程认证后的制度性空白。

## Recommended Reporting Wording

建议在正式报告中谨慎表述为：

```text
The robust 75-sample intersection between the CatBoost top-green subgroup and the RF proximity upper region appears to correspond to a Course_171-specific administrative subgroup. Course_171 maps to Design de Animação e Multimédia rather than a derivative of Course_17. A plausible institutional explanation is that these students entered through a CTeSP-to-bachelor pathway with credit recognition, causing first-year curricular activity features to be zero while the students still graduate successfully.
```

中文表述：

```text
CatBoost top-green 子群与 RF proximity upper region 的 75 个交集样本，是当前最稳健的特殊子群。`Course_171` 并非此前推测的 Course_17 衍生编码，而是独立的动画与多媒体设计专业代码。该子群前两学期学业记录接近全 0 但最终毕业，可能反映了 CTeSP 升本路径中的学分豁免机制，而不是数据错误或异常关系样本。
```

## Caveats

这份 findings 记录的是当前业务解释假设。若要在最终报告中作为强结论使用，建议补充以下证据：

- 给出 `Course_171 / L171` 与 `Design de Animação e Multimédia` 的官方 DGES 或 IPP 来源链接。
- 在 75 个样本上明确统计 `Course_171` 占比、`Target=Graduate` 占比、semester activity 为 0 的比例。
- 检查 `Previous qualification`、`Application mode`、`Curricular units ... credited` 等列是否与 CTeSP / credit recognition 解释一致。
- 避免称其为“关系户”“名誉学生”或数据错误；当前更合理表述是“特殊行政路径/学分豁免子群”。

  最重要的结论是：这 75 人不只是 Course_171 + enrolled/grade=0，而是 Course_171 + Graduate + daytime attendance + 全部
  semester academic activity fields 为 0 的一批记录。这个更强地支持它是某种特殊数据机制/课程记录机制。

  • 结论：CatBoost 能把“Course171 + zero curricular 这种结构”分出来，但不能可靠地区分其中的 Graduate。

  我刚按 notebook 同样参数重跑 CatBoost 验证了。

  这 75 人的预测结果：

  真实：Graduate 75

  CatBoost 预测：
  Graduate 40
  Dropout 27
  Enrolled 8

  准确率：53.3%
  平均 P(Graduate)：0.371

  按 split 看更明显：

  train: 35 / 59 预测为 Graduate
  val:    3 / 8  预测为 Graduate
  test:   2 / 8  预测为 Graduate

  所以如果你问的是：

  > CatBoost 有没有把这 75 人当成一个可预测的 Graduate 子类？

  答案是：没有稳定做到。尤其在 val/test 上很弱。

  但如果你问的是：

  > CatBoost 的 leaf/path representation 有没有把这批人和普通 Graduate 分开？

  答案是：有。

  原因是这 75 人在特征空间里非常像一个机制组：

  Course=171
  所有 semester academic activity = 0
  Target=Graduate

  但问题是：同样的 Course=171 + zero curricular 结构里还有很多非 Graduate：

  Course171 + zero curricular 总计 180 人：

  Graduate 75
  Dropout 77
  Enrolled 28

  CatBoost 对这 180 人的整体预测：

  真实 Dropout 77: 预测 Dropout 71
  真实 Enrolled 28: 预测 Enrolled 15
  真实 Graduate 75: 预测 Graduate 40

  也就是说，CatBoost 更容易把这类 zero-curricular 机制识别成 Dropout/非成功风险结构，但里面有一批真实标签是 Graduate，于是
  模型会犹豫。

  所以最终判断：

  CatBoost 分出了“Course171 zero-curricular 机制组”，但没有成功分出“这个机制组里的 Graduate 子类”。

  这也解释了为什么 t-SNE 图上它们很漂亮地聚在一起：t-SNE 展示的是 CatBoost leaf/path 相似性，不等于模型最终预测它们都是
  Graduate。图上的绿色是我们用真实标签上色，不是模型预测颜色。

  我把验证结果保存到了：

  visualization/outputs/catboost_selected75_prediction_check.csv

  如果要继续做，这批人应该作为一个独立机制组分析，而不是直接说 CatBoost 已经学会了一个可靠的 Graduate subtype。
