import os
from langchain_core.runnables import ConfigurableField
from langchain_google_vertexai import ChatVertexAI
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from typing import Any
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)
from langchain_google_community import VertexAISearchRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables import RunnablePassthrough
from google.genai import types
from app.schemas.chat_schema import PassiveGoal, Judgement, State


class LangGraph:
    def __init__(self):
        self.model = ChatVertexAI(model_name="gemini-2.0-flash-001", project=os.environ.get('GOOGLE_CLOUD_PROJECT'), location="us-central1")
        self.llm = self.model.configurable_fields(max_output_tokens=ConfigurableField(id='max_output_tokens'))
        self.ROLES = {
            "1": {
                "name": "スーパーバイザー",
                "description": "ユーザーの要望に対して、最適な回答を提供するための全体的な監督と調整を行う役割です。",
                "details": """LIXILの社員からの質問に対し、スーパーバイザーの目線から回答を行ってください。回答は、LIXILのデータソースのみを使用し生成してください。"""
            }
        }   # 回答を生成するAIの役割を記載する
        self.retriever = VertexAISearchRetriever(
            project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
            location_id="global",
            data_store_id=os.environ["DATASTORE"],
            engine_data_type=0,
            max_documents=5,
        )


    def goal_node(self, state: State) -> dict[str, Any]:
        query = state.query

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""
                    ユーザーの質問とユーザーの会話履歴を組み合わせて分析し、明確なユーザーのニーズを生成してください。
                    ニーズは、質問と会話履歴の内容に基づいて具体的である必要があります。
                    要件：
                    1. ニーズは明確である必要があります。
                    2. 以下の手順に従い回答の生成を行ってください。
                        - 会話履歴と質問を組み合わせて分析する。
                        - ユーザーのニーズを生成する。

                    4. 決して2.以外の行動を取ってはいけません。

                    """.strip()
                ),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        chain = prompt | self.llm.with_structured_output(PassiveGoal)
        result: PassiveGoal = chain.invoke({"query": query})
        return {
            "passive_goal": result.user_needs,
        }

    def selection_node(self, state: State) -> dict[str, Any]:
        query = state.passive_goal
        role_options = "\n".join([f"{k}. {v['name']}: {v['description']}" for k, v in self.ROLES.items()])

        prompt = ChatPromptTemplate.from_template(
            """要望を分析し、最も適切な回答担当ロールを選択してください。

            選択肢:
            {role_options}

            回答は選択肢の番号（1）のみを返してください。

            要望: {query}
            """.strip()
        )

        chain = prompt | self.llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
        role_number = chain.invoke({"role_options": role_options, "query": query}).strip()

        selected_role = self.ROLES.get(role_number, self.ROLES["1"])["name"]

        return {
            "current_role": selected_role
        }

    def answering_node(self, state: State) -> dict[str, Any]:
        query = state.query
        role = state.current_role
        role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in self.ROLES.values()])

        system_prompt = """
        あなたは{role}として回答してください。あなたの役割に基づいてユーザーのニーズを満たす回答を提供してください。
        提供されたcontextのみを使用して、この質問に答えてください。
        役割の詳細に従って回答を生成してください。
        役割の詳細:
        {role_details}
        ユーザーが求めているニーズは以下の通りです。
        {needs}
        """
        system_prompt = system_prompt.format(
            role=role,
            role_details=role_details,  # 修正: role_detailsをformatに追加
            needs=state.passive_goal,
        )

        # 事前にRetrieverで検索
        docs = self.retriever.get_relevant_documents(state.passive_goal)
        context = "\n".join([doc.page_content for doc in docs])
        # ドキュメントの内容やメタデータをgrounding_dataとして返す
        grounding_data = [{"uri": doc.metadata.get("source")} for doc in docs]

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt.strip()),
                HumanMessagePromptTemplate.from_template("質問：{question}, Context:{context}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": query, "context": context})

        return {
            "messages": [answer],
            "grounding_data": grounding_data
        }

    def check_node(self, state: State) -> dict[str, Any]:
        query = state.query
        answer = state.messages[-1] if state.messages else ""

        prompt = ChatPromptTemplate.from_template(
            """以下の回答の品質をチェック'False'または'True'を回答してください。
            品質をチェック項目を全て満たしている場合のみ'True'を回答することができます。
                1. 回答の日本語に問題がない
                2. 回答が空文字でない
                3. ハルシネーションが'False'である
            また、その判断理由も説明してください。

            ユーザーからの質問: {query}
            回答: {answer}
            ハルシネーション: {is_hallucination}
            """.strip()
        )

        chain = prompt | self.llm.with_structured_output(Judgement)

        result: Judgement = chain.invoke({"query": query, "answer": answer, "is_hallucination": state.is_hallucination})
        if result is None:
            raise ValueError("LLM の応答が無効です")

        return {
            "current_judge": result.judge,
            "judgement_reason": result.reason,
            "retry_count": state.retry_count + 1
        }

    def langgraph(self):
        workflow = StateGraph(State)

        # ノードを追加
        workflow.add_node("goal_create", self.goal_node)
        workflow.add_node("selection", self.selection_node)
        workflow.add_node("answering", self.answering_node)
        workflow.add_node("check", self.check_node)

        # # フローを設定
        workflow.set_entry_point("goal_create")
        workflow.add_edge("goal_create", "selection")
        workflow.add_edge("selection", "answering")
        workflow.add_edge("answering", "check")

        # checkノードの条件分岐
        workflow.add_conditional_edges(
            "check",
            lambda state: state.current_judge or state.retry_count >= 3,
            {True: END, False: "selection"}
        )

        checkpointer = MemorySaver()

        # グラフをコンパイル
        compiled = workflow.compile(checkpointer=checkpointer)

        return compiled
    