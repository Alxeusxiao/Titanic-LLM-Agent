# Titanic LLM Agent 数据分析

基于 DeepSeek LLM 的智能体（Agent）项目，通过 Function Calling 机制自动分析 Titanic 数据集，包括统计摘要、数据可视化和机器学习建模。

## 架构概览

```
User Prompt → LLM (DeepSeek) → Tool Call → 执行函数 → 返回结果 → LLM 综合回答
```

LLM 作为"大脑"进行决策和规划，通过调用工具函数完成具体任务，最终将结果汇总为自然语言回答。

## 功能 / 可用工具

| 工具 | 功能 | 输出 |
|------|------|------|
| `get_summary` | 数据统计摘要（均值、方差、生还分布） | 文本 |
| `create_plot` | 数据可视化 | `survived.png` / `age.png` |
| `train_model` | 随机森林建模预测生还率 | 测试集准确率 |

## 环境要求

- Python 3.8+
- DeepSeek API Key

### Python 依赖

```bash
pip install openai pandas matplotlib seaborn scikit-learn
```

## 快速开始

1. **设置 API Key**

```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "your-api-key-here"
```

2. **确保数据文件存在**

   `titanic_cleaned.csv` 应位于项目根目录。

3. **运行**

```bash
python main.py
```

## 运行示例

```
User> 请帮我分析 Titanic 数据，包括统计摘要、生还分布图和建模。

--- 模型调用工具 ---
工具: get_summary
参数: {}

--- 模型调用工具 ---
工具: create_plot
参数: {'chart': 'survived'}

--- 模型调用工具 ---
工具: train_model
参数: {}

Model> 分析结果如下：
1. 数据共 891 条记录，生还率约 38%...
2. 生还分布图已保存...
3. 随机森林模型准确率 0.82...
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `survived.png` | 生还/未生还分布柱状图 |
| `age.png` | 乘客年龄分布直方图 |

## 代码结构

```
main.py
├── LLM 初始化          # DeepSeek 客户端配置
├── 数据加载            # 读取 titanic_cleaned.csv
├── 工具函数
│   ├── get_summary()   # pandas 统计摘要
│   ├── create_plot()   # matplotlib 可视化
│   └── train_model()   # sklearn 随机森林
├── Tool Schema 定义    # OpenAI Function Calling 格式
├── send_messages()     # LLM 调用封装
└── Agent 主循环        # 自动调用工具直到任务完成
```

## 自定义任务

修改 `main.py` 中第 133 行的 `messages` 即可更换分析任务，例如：

```python
messages = [
    {"role": "user", "content": "只画年龄分布图"}
]
```