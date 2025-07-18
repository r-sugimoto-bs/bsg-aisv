import os
import re
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
from app.modules.firestore import FireStore


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
            data_store_id=os.getenv("FC_DATASTORE"),
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
                MessagesPlaceholder("history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
        chain = prompt | self.llm.with_structured_output(PassiveGoal)
        result: PassiveGoal = chain.invoke({"history": state.history, "query": query})
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
        # 事前にRetrieverで検索
        docs = self.retriever.get_relevant_documents(state.passive_goal)

        fake_id_map = {}
        context_parts = []
        #Geminiの回答生成で使用する出典のインデックスと引用内容の作成
        for i, doc in enumerate(docs, 1):
            fake_id = f"src_{i}"
            context_parts.append(f"[{fake_id}] {doc.page_content}")
            # 出典のインデックスと引用内容をマッピング
            fake_id_map[fake_id] = doc.metadata["source"]

            context = "\n\n".join(context_parts)

        system_prompt = """
        あなたは{role}として回答してください。あなたの役割に基づいてユーザーのニーズを満たす回答を提供してください。
        提供されたcontextのみを使用して、この質問に答えてください。
        回答には、使った情報の出典を [src_n] の形式で必ず明記してください（例: [src_1], [src_2]）。
        [src_num1, src_num2] のような記載は避けてください。（例：[src_3, src_4]ではなく[src_3][src_4]としてください)
        出典の一覧は書かないでください。
        役割の詳細に従って回答を生成してください。
        役割の詳細:
        {role_details}
        ユーザーが求めているニーズは以下の通りです。
        {needs}
        context:
        {context}

        """
        system_prompt = system_prompt.format(
            role=role,
            role_details=role_details,  # 修正: role_detailsをformatに追加
            needs=state.passive_goal if state.passive_goal else "",
            context=context if context else ""
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt.strip()),
                MessagesPlaceholder("history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"history": state.history, "question": query})
        # 出典情報を[n]のような形で表示する
        used_fake_ids = list(dict.fromkeys(re.findall(r"\[src_\d+\]", answer)))  # 順序保持
        final_map = {}
        remapped_answer = answer

        for new_idx, fake_id in enumerate(used_fake_ids, 1):
            num = str(new_idx)
            remapped_answer = remapped_answer.replace(fake_id, f"[{num}]")
            # 出典のインデックスから[]を取り除く（fake_id_mapは{src_1: hogehoge.pdf}のようになっている）
            clean_id = fake_id.strip("[]")
            # 対応する出典情報を記録
            final_map[num] = fake_id_map.get(clean_id, "unknown")

        return {
            "messages": [remapped_answer.strip()],
            "grounding_data": [final_map]
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

    def fetch_chat_node(self, state: State) -> dict[str, Any]:
        """チャット履歴を取得するノード"""
        firestore = FireStore()
        history_from_firestore = firestore.fetch_chat_log_to_input(state.user_id, state.session_id)
        history_to_langchain = []
        if not history_from_firestore:
            return {"history": []}
        else:
            for message in history_from_firestore:
                history_to_langchain.append(
                    HumanMessage(content=message.get("user-message"))
                )
                history_to_langchain.append(
                    AIMessage(content=message.get("agent-message"))
                )
            return {"history": history_to_langchain}

    def insert_chat_node(self, state: State) -> None:
        """チャット履歴をFirestoreに保存するノード"""
        firestore = FireStore()
        firestore.insert_current_chat(
            user_id=state.user_id,
            session_id=state.session_id,
            user_message=state.query,
            agent_message=state.messages[-1] if state.messages else ""
        )


    def langgraph(self):
        workflow = StateGraph(State)

        # ノードを追加
        workflow.add_node("goal_create", self.goal_node)
        workflow.add_node("selection", self.selection_node)
        workflow.add_node("answering", self.answering_node)
        workflow.add_node("check", self.check_node)
        workflow.add_node("fetch_chat", self.fetch_chat_node)
        workflow.add_node("insert_chat", self.insert_chat_node)

        # # フローを設定
        workflow.set_entry_point("fetch_chat")
        workflow.add_edge("fetch_chat", "goal_create")
        workflow.add_edge("goal_create", "selection")
        workflow.add_edge("selection", "answering")
        workflow.add_edge("answering", "check")

        # checkノードの条件分岐
        workflow.add_conditional_edges(
            "check",
            lambda state: state.current_judge or state.retry_count >= 3,
            {True: "insert_chat", False: "selection"}
        )
        workflow.add_edge("insert_chat", END)

        checkpointer = MemorySaver()

        # グラフをコンパイル
        compiled = workflow.compile(checkpointer=checkpointer)

        return compiled

