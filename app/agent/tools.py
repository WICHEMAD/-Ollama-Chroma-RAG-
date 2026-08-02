"""Agent 工具定义"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from app.rag.retriever import get_retriever, Retriever
from langchain_core.tools import Tool


# ==================== 文档检索工具 ====================
class DocumentSearchTool:
    """文档检索工具"""

    @staticmethod
    def run(query: str) -> str:
        try:
            retriever: Retriever = get_retriever()
            docs = retriever.retrieve(query)

            if not docs:
                return "未找到相关文档"

            parts = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get('source', '未知来源')
                page = doc.metadata.get('page', '')
                score = doc.metadata.get('rerank_score', doc.metadata.get('similarity_score', ''))

                source_info = f"来源[{i}]: {source}"
                if page:
                    source_info += f", 页码: {page}"
                if score:
                    source_info += f", 相关性: {score:.4f}"

                content = doc.page_content.strip()
                if len(content) > 500:
                    content = content[:500] + "..."

                parts.append(f"{source_info}\n{content}")

            return "\n\n".join(parts)

        except Exception as e:
            return f"文档检索失败: {str(e)}"


# ==================== 计算器工具 ====================
class CalculatorTool:
    """计算器工具 - 执行数学计算"""

    @staticmethod
    def run(expression: str) -> str:
        """
        执行数学计算

        Args:
            expression: 数学表达式，如 "2+2"、"3*4-5" 等

        Returns:
            计算结果或错误信息
        """
        try:
            # ✅ 安全检查：只允许数字和基本运算符
            allowed_chars = set("0123456789.+-*/()%^ ")

            # 检查非法字符
            for char in expression:
                if char not in allowed_chars:
                    return f"错误：表达式包含非法字符 '{char}'"

            # 限制长度
            if len(expression) > 100:
                return "错误：表达式过长"

            # 执行计算
            result = eval(expression)
            return f"计算结果: {result}"

        except SyntaxError:
            return f"错误：表达式语法错误: {expression}"
        except ZeroDivisionError:
            return "错误：除以零"
        except Exception as e:
            return f"计算失败: {str(e)}"


# ==================== 工具注册（使用 LangChain Tool 类） ====================
document_search = Tool(
    name="document_search",
    func=DocumentSearchTool.run,
    description="""
用于在私有知识库中搜索与用户问题相关的文档片段。
当用户询问具体事实、数据、流程等需要从文档中查找的信息时使用此工具。
输入应为完整的查询语句。

示例：
- 输入: "陈冠霖是谁？"
- 输入: "项目开发流程是什么？"
"""
)

calculator = Tool(
    name="calculator",
    func=CalculatorTool.run,
    description="""
执行数学计算。输入一个数学表达式，返回计算结果。
支持的运算：加(+)、减(-)、乘(*)、除(/)、幂(^)、取模(%)、括号。

示例：
- 输入: "2+2"
- 输入: "3*4-5"
- 输入: "(10+5)*2"
- 输入: "2^3"
"""
)
# ✅ 创建工具实例供外部调用
document_search_tool = DocumentSearchTool()

TOOLS: List[Tool] = [document_search, calculator]

# def get_tool_by_name(name: str) -> Tool:
#     """根据名称获取工具"""
#     for tool in TOOLS:
#         if tool.name == name:
#             return tool
#     return None


# 使用示例
if __name__ == "__main__":
    tool = DocumentSearchTool()
    result = tool.run("陈冠霖是谁？")
    print("检索结果:")
    print(result)