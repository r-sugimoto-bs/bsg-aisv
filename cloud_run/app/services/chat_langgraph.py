import os
import re
from langchain_core.runnables import ConfigurableField
from langchain_google_vertexai import ChatVertexAI
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from typing import Any
from difflib import get_close_matches
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
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from app.schemas.chat_schema import PassiveGoal, Judgement, State
from app.modules.firestore import FireStore

import json
import requests
import google.auth
from google.auth.transport.requests import Request

class LangGraph:
    def __init__(self):
        self.model = ChatVertexAI(model_name="gemini-2.0-flash-001", project=os.environ.get('GOOGLE_CLOUD_PROJECT'), location="us-central1")
        self.llm = self.model.configurable_fields(max_output_tokens=ConfigurableField(id='max_output_tokens'))
        self.ROLES = {
            "1": {
                "name": "スーパーバイザー",
                "description": "ユーザーの要望に対して、最適な回答を提供するための全体的な監督と調整を行う役割です。",
                "details": """
                LIXILの若手スーパーバイザーの社員からの質問に対し、ベテランスーパーバイザーの目線で回答を行ってください。
                基本的には対話を重視して話が長くなりすぎないように心がけてください。
                アドバイスするときはその加盟店がある場所の地域的な特性や、その加盟店の売り上げ、イベント状況を加味して具体的なアクションプランを提示してください。
                回答は、LIXILのデータソースのみを使用し生成してください。
                """
            }
        }   # 回答を生成するAIの役割を記載する
        self.retriever = VertexAISearchRetriever(
            project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
            location_id="global",
            data_store_id=os.getenv("DATA_STORE_ID"),
            engine_data_type=0,
            max_documents=5,
        )
        #ToDo Firestoreから店舗情報は読み取る
        self.branch_store_map = {
            "東京支社":  ["ハウステックス", "ウスクラ建設", "善光建設", "エヌティーサービス", "ライファ大塚","その他"]
        }
        self.branch_list = list(self.branch_store_map.keys())
        self.project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.location = "global"  # "us" または "eu" の場合もあり
        self.engine_id = os.getenv("FC_DATASTORE") # Engine ID (App ID) を環境変数から取得
        # Discovery Engine クライアントの初期化
        client_options = (
            ClientOptions(api_endpoint=f"{self.location}-discoveryengine.googleapis.com")
            if self.location != "global"
            else None
        )
        self.discovery_client = discoveryengine.ConversationalSearchServiceClient(
            client_options=client_options
        )

        # Serving Config のフルパスを作成
        self.serving_config = ( f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/engines/{self.engine_id}/servingConfigs/default_serving_config"
            )

    def fetch_node(self, state: State) -> dict:
        """【開始】会話履歴と最新Stateを読み込む"""
        firestore = FireStore()
        
        # 1. 会話履歴を読み込み、LangChainのMessage形式に変換
        raw_history = firestore.fetch_chat_log_to_input(state.user_id, state.session_id)
        messages = []
        for log in raw_history:
            messages.append(HumanMessage(content=log.get("user-message", "")))
            if log.get("agent-message"):
                messages.append(AIMessage(content=log.get("agent-message", "")))
        state.history = messages
        
        # 2. 最新のState情報を読み込んで反映
        latest_state_data = firestore.fetch_latest_state_from_history(state.user_id, state.session_id)
        response = {"history": messages}
        response.update(latest_state_data)
        return response

    def save_node(self, state: State) -> dict:
        """【終了】会話と最新Stateをマージして保存する"""
        firestore = FireStore()
        firestore.insert_chat_with_state(state)
        return {}

    def capture_and_history_node(self, state: State) -> dict:
        """ユーザーの入力を履歴に追加し、必要ならスロットを埋める"""
        # このノードで更新される可能性がある値を準備
        updated_history = state.history + [HumanMessage(content=state.query)]
        updated_store_name = state.store_name
        updated_branch_name = state.branch_name
        updated_asking_slot = state.asking_slot

        if state.asking_slot == "store_name":
            updated_store_name = state.query.strip()
            updated_asking_slot = None
            for branch, stores in self.branch_store_map.items():
                if updated_store_name in stores:
                    updated_branch_name = branch
                    break
                
        return {
            "history": updated_history,
            "store_name": updated_store_name,
            "branch_name": updated_branch_name,
            "asking_slot": updated_asking_slot,
        }
    
    def extract_entities_node(self, state: State) -> dict:
        """
        1) Python の直接部分一致 or get_close_matches で store_name を確実に拾う
        2) 取れなかったら Gemini に抽出を頼む
        """
        text = state.query.strip()
        # 全店舗のフラットリスト
        all_stores = sum(self.branch_store_map.values(), [])

        # ① 直接部分一致
        for branch, stores in self.branch_store_map.items():
            for store in stores:
                if store in text:
                    state.store_name = store
                    state.branch_name = branch
                    print(f"[EXTRACT] direct match: store={store}, branch={branch}")
                    return {
                        "store_name": state.store_name,
                        "branch_name": state.branch_name,
                    }

        # ② fuzzy match
        fuzzy = get_close_matches(state.query, all_stores, n=1, cutoff=0.6)
        if fuzzy:
            store = fuzzy[0]
            state.store_name = store
            for branch, stores in self.branch_store_map.items():
                if store in stores:
                    state.branch_name = branch
            print(f"[DEBUG extract] fuzzy match → store_name={store!r}, branch_name={state.branch_name!r}")
            return {}
        # LLM フォールバック
        print("[DEBUG extract] falling back to LLM")

        # ③ 最終手段で LLM に聞く
        stores_str = "\n".join(f"- {s}" for s in all_stores)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "次の店舗一覧の中から、質問文に“完全一致”で含まれる"
                "店舗名を一つだけ返してください。含まれなければその他を選択してください。"
            )),
            HumanMessagePromptTemplate.from_template(
                "店舗一覧:\n{stores}\n\n質問:\n{query}"
            ),
        ])
        chain = prompt | self.llm | StrOutputParser()
        extracted = chain.invoke({"stores": stores_str, "query": text}).strip()
        print(f"[EXTRACT] LLM fallback returned: {extracted!r}")

        if extracted in all_stores:
            state.store_name = extracted
            for branch, stores in self.branch_store_map.items():
                if extracted in stores:
                    state.branch_name = branch
                    print(f"[EXTRACT] LLM matched branch={branch}, store={extracted}")
                    break

        return {}
    
    # ② スロット未入力チェック（どちらか一方が埋まっていれば OK）
    def slot_check_node(self, state: State) -> dict:
        # store_name が未設定なら質問
        if state.store_name is None:
            state.asking_slot = "store_name"
            return {"need_slot": "store_name"}
        # store_name が入っていればそのまま answer へ
        return {}

    def ask_slot_node(self, state: State) -> dict:
        """
        store_name が未設定のときに、AIに「どの店舗を知りたいか？」を聞かせるプロンプトで
        自動生成した質問文を返します。
        """
        # 1) プロンプト定義（お好きな内容に書き換えてください）
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            あなたは LIXILのAIスーパーバイザーです。
            若手社員の相談に対して一文でリアクションをして、 
            ユーザーがどの加盟店についての話をしたいのかを具体的に教えてもらうための、丁寧な質問文を日本語で１つだけ生成してください。  
            """.strip()),
            MessagesPlaceholder("history"),
            HumanMessagePromptTemplate.from_template("{passive_goal}")
        ])

        # 2) チェーン組み立て
        chain = prompt | self.llm | StrOutputParser()

        # 3) 実行
        question = chain.invoke({
            "history": state.history,
            "passive_goal": state.passive_goal
        }).strip()

        return {"messages": [question]}

    def goal_node(self, state: State) -> dict[str, Any]:
        """会話履歴全体を基に、LLMにユーザーの最終的なニーズを生成させる"""
        print(f"INFO: Generating goal with history of {len(state.history)} messages.")
        
        # 常に履歴全体をLLMに渡してニーズを生成する
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="ユーザーとの過去の会話履歴全体と最新の質問を考慮して、ユーザーが本当に求めている具体的なニーズ（ゴール）を1つ、簡潔な日本語で生成してください。"),
            MessagesPlaceholder("history"),
        ])
        chain = prompt | self.llm.with_structured_output(PassiveGoal)
        result: PassiveGoal = chain.invoke({"history": state.history})
        
        print(f"✅ Generated Goal: {result.user_needs}")
        return {"passive_goal": result.user_needs}

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


    # def answering_node(self, state: State) -> dict[str, Any]:
    #     # 環境変数読み込み
    #     project_id      = os.environ["GOOGLE_CLOUD_PROJECT"]
    #     location        = os.environ.get("LOCATION", "global")
    #     engine_id       = os.environ["FC_DATASTORE"]  # 統合検索アプリの Engine ID
    #     # 確定済みのデータストアIDをカンマ区切りで設定
    #     datastore_ids   = os.environ.get("BLENDED_DATASTORES", "").split(",")

    #     # クライアント初期化
    #     client_options = (
    #         ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
    #         if location != "global" else None
    #     )
    #     client = discoveryengine.ConversationalSearchServiceClient(
    #         client_options=client_options
    #     )

    #     # serving_config のパス（engine_id ベース）
    #     serving_config = (
    #         f"projects/{project_id}/locations/{location}/collections/"
    #         f"default_collection/engines/{engine_id}/servingConfigs/"
    #         "default_serving_config"
    #     )

    #     # クエリ準備
    #     query = discoveryengine.Query(text=state.passive_goal or state.query)

    #     # Query フェーズのオプション
    #     query_understanding_spec = discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
    #         query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
    #             disable=False,
    #             max_rephrase_steps=1
    #         )
    #     )

    #     # 統合検索時に特定のデータストアのみを検索するための設定
    #     data_store_specs = [
    #         discoveryengine.AnswerQueryRequest.SearchSpec.DataStoreSpec(
    #             data_store=f"projects/{project_id}/locations/{location}/dataStores/{ds.strip()}"
    #         )
    #         for ds in datastore_ids if ds.strip()
    #     ]

    #     # Search フェーズのオプション（データストア指定 + 件数制限）
    #     search_spec = discoveryengine.AnswerQueryRequest.SearchSpec(
    #         search_params=discoveryengine.AnswerQueryRequest.SearchSpec.SearchParams(
    #             max_return_results=3
    #         ),
    #         data_store_specs=data_store_specs
    #     )

    #     # Answer フェーズのオプション
    #     answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
    #         ignore_adversarial_query=False,
    #         ignore_non_answer_seeking_query=False,
    #         ignore_low_relevant_content=False,
    #         model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
    #             model_version="gemini-2.0-flash-001/answer_gen/v1"
    #         ),
    #         prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
    #             preamble=(
    #                 "あなたはベテランのAIスーパーバイザーです。以下の役割を厳守してください。\n"
    #                 "- 検索結果のみを情報源として、ユーザーの質問に答える\n"
    #                 "- 若手の考えを引き出すことと、RAGで取得した抽象的な情報を具体的なアクションプランとして伝えることを意識する\n"
    #                 "- 対話を意識し、回答が長くなりすぎないようにする\n"
    #                 f"- 支社『{state.branch_name or '未指定'}』や店舗『{state.store_name or '未指定'}』の情報を考慮に入れる"
    #             )
    #         ),
    #         include_citations=True,
    #     )

    #     # リクエスト組み立て
    #     request = discoveryengine.AnswerQueryRequest(
    #         serving_config=serving_config,
    #         query=query,
    #         session=None,
    #         query_understanding_spec=query_understanding_spec,
    #         search_spec=search_spec,
    #         answer_generation_spec=answer_generation_spec,
    #     )

    #     # API 呼び出し
    #     response = client.answer_query(request=request)

    #     # レスポンス整形
    #     resp = discoveryengine.AnswerQueryResponse.to_dict(response)
    #     answer_text = resp.get("answer", {}).get("text", "回答が見つかりませんでした。")

    #     # 引用マッピング
    #     source_map = {
    #         str(i+1): ref.get("chunk_info", {})
    #                         .get("document_metadata", {})
    #                         .get("uri", "不明なソース")
    #         for i, ref in enumerate(resp.get("answer", {}).get("references", []))
    #     }
    #     used = sorted(set(re.findall(r"\[(\d+)\]", answer_text)), key=int)
    #     grounding = { num: source_map.get(num, "不明なソース") for num in used }

    #     return {
    #         "messages": [answer_text.strip()],
    #         "grounding_data": [grounding]
    #     }

    def answering_node(self, state: State) -> dict[str, Any]:
        # 環境変数読み込み
        project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        location   = os.environ.get("LOCATION", "global")
        engine_id  = os.environ["FC_DATASTORE"]
        query = state.query
        role = state.current_role
        role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in self.ROLES.values()])
        fake_id_map = {}
        
        # ADC で認証情報を取得 (Cloud Run 環境のサービスアカウントを利用)
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        token = creds.token

        # URL 組み立て (v1alpha default_search:answer エンドポイント)
        url = (
            f"https://discoveryengine.googleapis.com/v1alpha/"
            f"projects/{project_id}/locations/{location}/collections/default_collection/"
            f"engines/{engine_id}/servingConfigs/default_search:answer"
        )

        # リクエストボディ作成
        body = {
            "query": {"text": state.passive_goal or state.query, "queryId": ""},
            "session": "",
            "relatedQuestionsSpec": {"enable": True},
            "answerGenerationSpec": {
                "ignoreAdversarialQuery": False,
                "ignoreNonAnswerSeekingQuery": False,
                "ignoreLowRelevantContent": False,
                "multimodalSpec": {},
                "includeCitations": True,
                "promptSpec": {
                    "preamble": (
                        "あなたはベテランのAIスーパーバイザーです。以下の役割を厳守してください。\n"
                        "- 検索結果のみを情報源として、ユーザーの質問に答える\n"
                        "- 該当店舗だけではなく店舗でうまくいっているようなことがあれば参考情報として回答してください\n"
                        "- 既存の情報を整理し、具体的なアクションプランをねん出することにフォーカスしてください"
                        f"- 支社『{state.branch_name or '未指定'}』や店舗『{state.store_name or '未指定'}』の情報を考慮に入れる"
                    )
                },
                # モデルには 'stable' を使う
                "modelSpec": {"modelVersion": "stable"}
            }
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()

        # answerText 抽出と整形
        raw_text = result.get("answer", {}).get("answerText", "回答が見つかりませんでした。")
        print(f'===================={raw_text}')
        summary = re.sub(r"\r", "", raw_text).strip()

        # 追加: 生成AIで要約を会話用に再構成
        system_prompt = """
                あなたは若手社員の相談に真摯に対応するAIスーパーバイザーです。
                以下の役割に基づいて若手社員に返答する質問を考えてください。
    
                【役割】
                - 若手社員の質問に対して別の生成AIが作成したcontextの内容を参考にして、この質問に答えてください。
                - 役割の詳細に従って回答を生成してください。
                - 役割の詳細:
                - {role_details}
                - ユーザーが求めているニーズは以下の通りです。
                - {needs}
                - context:
                - {context}
                
                【回答ルール】
                - contextの内容はあくまで回答例でありあなた個人の意見として考えてください。
                - 日本語で回答してください。
                - 抽象的な内容ではなくてcotextの内容から取得した具体的な内容を踏まえて回答してください。
                    - 例えば地域特性に合わせた内容にする。（例：近隣の住宅事情やニーズに合わせたリフォーム事例を紹介する）のような内容ではなく、その加盟店があるであろう地域の特性を考慮して具体的にどんな内容にすべきかあなた自身が推論してください。
                - 対話を意識し、必要じゃない時は200字以内で回答を生成してください。
                - 質問者は加盟店の人間ではなく若手SVに過ぎないのであなたが連携された情報以上にその加盟店については知らないことが多いのでどんなことが課題だと思いますかなど抽象的な質問をしないでください。
                - あなたはベテランのAIスーパーバイザーなので、若手社員に方針の質問丸投げせずにcontextから取得した抽象的な情報を自分の意見として具体的なアクションプランとして伝えることを意識してください
                - また支社や店舗の情報をもとに地域性を考慮した情報をあなた自身が考えて回答してください。
       
            """
        if state.branch_name:
            system_prompt += f"・支社: {state.branch_name}\n"
        if state.store_name:
            system_prompt += f"・店舗: {state.store_name}\n"
            
        system_prompt = system_prompt.format(
            role=role,
            role_details=role_details,
            needs=state.passive_goal if state.passive_goal else "",
            context=summary if summary else ""
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

        source_map = {}
        for i, ref in enumerate(result.get("answer", {}).get("references", []), start=1):
            if "chunkInfo" in ref:
                uri = ref["chunkInfo"]["documentMetadata"].get("uri", "")
            else:
                uri = ref.get("structuredDocumentInfo", {}).get("document", "")
            source_map[str(i)] = uri
        used = sorted(set(re.findall(r"\[(\d+)\]", answer)), key=int)
        grounding = {num: source_map.get(num, "") for num in used}

        return {
            "messages": [answer.strip()],
            "grounding_data": [grounding]
        }
        
    # def answering_node(self, state: State) -> dict[str, Any]:
    #     query = state.query
    #     role = state.current_role
    #     role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in self.ROLES.values()])
    #     # 事前にRetrieverで検索
    #     docs = self.retriever.get_relevant_documents(state.passive_goal)

    #     fake_id_map = {}
    #     context_parts = []
    #     #Geminiの回答生成で使用する出典のインデックスと引用内容の作成
    #     for i, doc in enumerate(docs, 1):
    #         fake_id = f"src_{i}"
    #         context_parts.append(f"[{fake_id}] {doc.page_content}")
    #         # 出典のインデックスと引用内容をマッピング
    #         fake_id_map[fake_id] = doc.metadata["source"]

    #         context = "\n\n".join(context_parts)
    #     system_prompt = """
    #     あなたは{role}として回答してください。あなたの役割に基づいてユーザーのニーズを満たす回答を提供してください。
    #     提供されたcontextのみを使用して、この質問に答えてください。
    #     回答には、使った情報の出典を [src_n] の形式で必ず明記してください（例: [src_1], [src_2]）。
    #     [src_num1, src_num2] のような記載は避けてください。（例：[src_3, src_4]ではなく[src_3][src_4]としてください)
    #     出典の一覧は書かないでください。
    #     役割の詳細に従って回答を生成してください。
    #     役割の詳細:
    #     {role_details}
    #     ユーザーが求めているニーズは以下の通りです。
    #     {needs}
    #     context:
    #     {context}
        
    #     日本語で回答してください。
    #     対話を意識して回答文が長くなりすぎないようにしてください。
    #     あなたはベテランのAIスーパーバイザーなので、若手の考えを引き出しつつも質問丸投げせずに
    #     データソースから取得した抽象的な情報を自分の意見として具体的なアクションプランとして伝えることを意識してください
    #     また支社や店舗の情報をもとに地域に特化した情報を含めて考えてください。
    #     """
        
        # if state.branch_name:
        #     system_prompt += f"・支社: {state.branch_name}\n"
        # if state.store_name:
        #     system_prompt += f"・店舗: {state.store_name}\n"
        
        # system_prompt = system_prompt.format(
        #     role=role,
        #     role_details=role_details,  # 修正: role_detailsをformatに追加
        #     needs=state.passive_goal if state.passive_goal else "",
        #     context=context if context else ""
        # )
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         SystemMessage(
        #             content=system_prompt.strip()),
        #         MessagesPlaceholder("history"),
        #         HumanMessagePromptTemplate.from_template("{question}"),
        #     ]
        # )

        # chain = prompt | self.llm | StrOutputParser()
        # answer = chain.invoke({"history": state.history, "question": query})
    #     # 出典情報を[n]のような形で表示する
        # used_fake_ids = list(dict.fromkeys(re.findall(r"\[src_\d+\]", answer)))  # 順序保持
        # final_map = {}
        # remapped_answer = answer

        # for new_idx, fake_id in enumerate(used_fake_ids, 1):
        #     num = str(new_idx)
        #     remapped_answer = remapped_answer.replace(fake_id, f"[{num}]")
        #     # 出典のインデックスから[]を取り除く（fake_id_mapは{src_1: hogehoge.pdf}のようになっている）
        #     clean_id = fake_id.strip("[]")
        #     # 対応する出典情報を記録
        #     final_map[num] = fake_id_map.get(clean_id, "unknown")

        # return {
        #     "messages": [remapped_answer.strip()],
        #     "grounding_data": [final_map]
        # }

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
        wf = StateGraph(State)
        wf.add_node("fetch", self.fetch_node)
        wf.add_node("capture", self.capture_and_history_node)
        wf.add_node("extract", self.extract_entities_node)
        wf.add_node("goal", self.goal_node)
        wf.add_node("select", self.selection_node)
        wf.add_node("slot_check", self.slot_check_node)
        wf.add_node("ask_slot", self.ask_slot_node)
        wf.add_node("answer", self.answering_node)
        wf.add_node("check", self.check_node)
        wf.add_node("save", self.save_node)

        # グラフの繋がりを定義
        wf.set_entry_point("fetch")
        wf.add_edge("fetch", "capture")
        wf.add_edge("capture", "extract")
        wf.add_edge("extract", "goal")
        wf.add_edge("goal", "select")
        wf.add_edge("select", "slot_check")

        wf.add_conditional_edges(
            "slot_check",
            lambda s: s.store_name is None,
            {True: "ask_slot", False: "answer"}
        )
        
        wf.add_edge("ask_slot", "save")
        wf.add_edge("answer", "check")

        wf.add_conditional_edges(
            "check",
            lambda s: s.current_judge or s.retry_count >= 3,
            {True: "save", False: "select"}
        )
        
        wf.add_edge("save", END)

        return wf.compile(checkpointer=MemorySaver())
