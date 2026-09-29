# 本地量子计算实验台

Python/FastAPI 执行状态向量计算，React/Vite 编辑线路、逐步运行、查看测量坍缩和重复实验统计；全程本地运行，不依赖量子云服务。

## 已实现能力

- 1–8 个量子位的本地状态向量模拟。
- 初始计算态、线路版本和随机种子由后端持久化到 `data/` JSON 文件。
- 单比特门：`X`、`Y`、`Z`、`H`、`S`、`T`、`RX`、`RY`、`RZ`。
- 双比特门：`CNOT`、`SWAP`。
- `MEASURE` 在后端按当前概率采样、坍缩，并让坍缩后的状态影响后续步骤。
- 非法量子位、重复步骤、缺少角度、CNOT/SWAP 自连等错误会阻止开始运行和保存。
- 重复实验统计可核对单比特叠加的 50/50，以及 Bell 线路只出现 `00`、`11`。

## 启动后端

```bash
cd backend
PYTHONPATH=. ../.venv/bin/python -m uvicorn quantum_lab.app:app --host 127.0.0.1 --port 8000
```

## 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开 Vite 输出的本地地址。前端通过代理访问 `/api`。

## 测试

```bash
cd backend
PYTHONPATH=. ../.venv/bin/python -m pytest
cd ../frontend
npm run build
```

## 后续扩展

噪声模型可在 `backend/quantum_lab/simulator.py` 的门应用和测量流程中加入 Kraus/密度矩阵通道；算法实验可新增后端实验接口，继续保持前端只提交线路、后端返回计算结果。
