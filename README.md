# trash-can RAG项目

## 项目结构说明
- `main.py`：项目入口，主运行逻辑
- `data_processor.py`：文档加载、文本切分、数据预处理
- `retriever.py`：向量检索模块，负责召回相关知识库内容
- `generator.py`：大模型生成回答模块
- `memory_manager.py`：对话历史、上下文记忆管理
- `evaluator2.py`：回答效果评估模块
- `desktop_app.py`：桌面端GUI应用入口
