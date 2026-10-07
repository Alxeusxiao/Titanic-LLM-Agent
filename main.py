from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json

# ========== 初始化 LLM ==========
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ========== 加载数据 ==========
df = pd.read_csv("titanic_cleaned.csv")

# ========== 工具函数 ==========
def get_summary():
    stats = df.describe().to_string()
    survive_cnt = df["Survived"].value_counts().to_string()
    return f"数值统计摘要:\n{stats}\n\nSurvived 分布:\n{survive_cnt}"

def create_plot(chart: str = "survived"):
    # ✅ Windows 下最稳定的中文字体配置
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    if chart == "survived":
        counts = df["Survived"].value_counts()
        plt.figure(figsize=(6, 4))
        bars = plt.bar(["未生还", "生还"], counts.values, color=["#4C72B0", "#DD8452"])
        plt.title("生还情况分布")
        plt.ylabel("人数")
        
        # 在柱子上标数字
        for i, v in enumerate(counts.values):
            plt.text(i, v + 5, str(v), ha="center")
        
        plt.tight_layout()
        plt.savefig("survived.png", dpi=300, bbox_inches="tight")
        plt.close()
        return "生还分布图已保存为 survived.png"

    if chart == "age":
        plt.figure(figsize=(6, 4))
        plt.hist(df["Age"].dropna(), bins=30, color="#55A868", edgecolor="black")
        plt.title("年龄分布")
        plt.xlabel("年龄")
        plt.ylabel("人数")
        plt.tight_layout()
        plt.savefig("age.png", dpi=300, bbox_inches="tight")
        plt.close()
        return "年龄分布图已保存为 age.png"

    return "未知图表类型"

def train_model():
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features].copy()

    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    le = LabelEncoder()
    X["Sex"] = le.fit_transform(X["Sex"])
    X["Embarked"] = le.fit_transform(X["Embarked"])

    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    return f"随机森林模型训练完成，测试集准确率: {acc:.4f}"

# ========== Tool Schema ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "对 Titanic 数据进行统计摘要，包括均值、方差、Survived 分布等",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": "绘制 Titanic 数据可视化图表",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart": {
                        "type": "string",
                        "enum": ["survived", "age"],
                        "description": "图表类型：survived（生还分布）或 age（年龄分布）",
                    }
                },
                "required": ["chart"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": "使用 sklearn 训练随机森林模型预测 Survived",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ========== LLM 调用封装 ==========
def send_messages(messages, tools=None):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message

# ========== Agent 主流程 ==========
messages = [
    {"role": "user", "content": "请帮我分析 Titanic 数据，包括统计摘要、生还分布图和建模。"}
]

print("User>", messages[0]["content"], "\n")

while True:
    message = send_messages(messages, tools=tools)

    # 如果没有工具调用，直接输出回答
    if not message.tool_calls:
        print("Model>", message.content)
        break

    # 如果有工具调用
    messages.append(message)

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print("--- 模型调用工具 ---")
        print("工具:", name)
        print("参数:", args)

        if name == "get_summary":
            result = get_summary()
        elif name == "create_plot":
            result = create_plot(**args)
        elif name == "train_model":
            result = train_model()
        else:
            result = "未知工具"

        print("工具结果:", result, "\n")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })