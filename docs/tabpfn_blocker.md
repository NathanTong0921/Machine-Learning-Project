# TabPFN Blocker

## 当前状态

- `tabpfn` 包已经成功安装
- 运行 [run_tabpfn_baseline.py](/media/990Pro/ML/scripts/transformer/run_tabpfn_baseline.py) 时，阻塞在官方模型权重下载前的一次性 license 验证

## 具体报错

核心错误是：

- `tabpfn.errors.TabPFNLicenseError`

含义：

- 当前环境没有可交互终端/浏览器
- TabPFN 需要先在官方页面接受 license
- 然后通过 `TABPFN_TOKEN` 提供 token 才能下载本地推理权重

## 官方继续方式

1. 打开：

```text
https://ux.priorlabs.ai
```

2. 登录并在 `Licenses` 页面接受 license
3. 在账户页面复制 API Key
4. 设置环境变量：

```bash
export TABPFN_TOKEN="<your-api-key>"
```

然后重新运行：

```bash
python scripts/transformer/run_tabpfn_baseline.py
```

## 当前判断

- 这不是项目代码实现问题
- 也不是依赖安装失败
- 是官方模型下载机制要求的认证步骤

## 已完成部分

- `tabpfn` 已安装
- baseline 脚本已写好：
  - [run_tabpfn_baseline.py](/media/990Pro/ML/scripts/transformer/run_tabpfn_baseline.py)
- 说明文档已写好：
  - [run_tabpfn_baseline.md](/media/990Pro/ML/docs/run_tabpfn_baseline.md)
