from google import genai
from google.genai import types
import base64
from pydantic import BaseModel
import os
import json
from app.schemas.chat_schema import State, PassiveGoal, Judgement
from typing import Optional, Type, List, Dict, Any

ROLES = {
            "1": {
                "name": "スーパーバイザー",
                "description": "ユーザーの要望に対して、最適な回答を提供するための全体的な監督と調整を行う役割です。",
                "details": """LIXILの社員からの質問に対し、スーパーバイザーの目線から回答を行ってください。回答は、LIXILのデータソースのみを使用し生成してください。"""
            }
        }


class GeminiService:
    def __init__(self, project_id: Optional[str] = None, location: str = 'us-central1', model: str = "gemini-2.0-flash-001"):
        """
        GeminiService の初期化

        Args:
            project_id (Optional[str]): Google Cloud プロジェクト ID。環境変数 GOOGLE_CLOUD_PROJECT から取得する場合は None を指定。
            location (str): Vertex AI のロケーション。
            model (str): 使用する Gemini モデル名。
        """
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set or project_id must be provided.")

        self.location = location
        self.model = model
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        self.safety_settings = [
            types.SafetySetting(category=category, threshold="OFF")
            for category in [
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_HARASSMENT",
            ]
        ]

    def _create_generation_config(self, system_prompt: str, temperature: float = 0.0, top_p: float = 0.95,
                                 max_output_tokens: int = 8192, response_modalities: List[str] = ["TEXT"],
                                 response_mime_type: Optional[str] = None, response_schema: Optional[str] = None,
                                 tools:List = None, **kwargs: Any):
        """
        GenerateContentConfig を作成する共通関数

        Args:
            system_prompt (str): システムプロンプト。
            temperature (float): ランダム性 (0.0 が最も決定的)。
            top_p (float):  確率が高い上位トークンのセットから選択する確率の閾値。
            max_output_tokens (int): 最大出力トークン数。
            response_modalities (List[str]): レスポンスのモダリティ。
            response_mime_type (Optional[str]): レスポンスの MIME タイプ (JSON の場合 "application/json")。
            response_schema (Optional[str]): レスポンスのスキーマ (JSON の場合)。
            **kwargs: その他の GenerationConfig パラメータ (例: safety_settings)。

        Returns:
            GenerationConfig: 設定オブジェクト。
        """
        config_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "response_modalities": response_modalities,
            "safety_settings": self.safety_settings,
            "system_instruction": [types.Part.from_text(text=system_prompt)]
        }
        if response_mime_type:
            config_params["response_mime_type"] = response_mime_type
        if response_schema:
            config_params["response_schema"] = response_schema
        # toolsをPydanticやAPIが受け入れる形式（例: dictやlist of dict）に変換
        if tools:
            config_params["tools"] = [tool.to_dict() if hasattr(tool, "to_dict") else tool for tool in tools]

        # 個別のメソッド呼び出しでconfigを上書きできるようにする。
        config_params.update(kwargs)
        return types.GenerateContentConfig(**config_params)

    def _set_contents(self, message: str, image_data: Optional[str] = None):
        """
        コンテンツを設定する共通関数

        Args:
            message (str): ユーザーメッセージ。
            image_data (Optional[str]): Base64 エンコードされた画像データ (PNG)。

        Returns:
            List[Content]: コンテンツのリスト。
        """
        contents = []
        if image_data:
            try:
                # Base64 形式のデータをデコードし、定義してある形式に代入
                contents.append(types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type="image/png",
                ))
            except Exception as e:
                raise ValueError(f"Invalid image_data: {e}")

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=message)
                ]
            )
        )
        return contents
    
    def _fetch_grounding_data(self, response: types.GenerateContentResponse) -> List[Dict[str, str]]:
        """
        レスポンスからグラウンディングデータを抽出する

        Args:
            response (types.GenerateContentResponse): Gemini の生成コンテンツレスポンス。

        Returns:
            List[Dict[str, str]]: グラウンディングデータのリスト。
        """
        grounding_data = []
        for source in response.candidates[0].grounding_metadata.grounding_chunks:
                grounding_data.append(
                    {"title": source.retrieved_context.title,
                     "uri": source.retrieved_context.uri}
                )
        return grounding_data

    def generate_for_gemini_text(self, system_prompt: str, message: str,
                                 temperature: float = 0.0, top_p: float = 0.95, max_output_tokens: int = 8192,
                                 tools: List = None, **kwargs: Any) -> str:
        """
        Gemini でテキストを生成する

        Args:
            system_prompt (str): システムプロンプト。
            message (str): ユーザーメッセージ。
            temperature (float): ランダム性 (0.0 が最も決定的)。
            top_p (float):  確率が高い上位トークンのセットから選択する確率の閾値。
            max_output_tokens (int): 最大出力トークン数。
            **kwargs: その他の GenerateContentConfig パラメータ (例: safety_settings)。

        Returns:
            str: 生成されたテキスト。
        """
        grounding_data = None
        contents = self._set_contents(message)
        config = self._create_generation_config(
            system_prompt, temperature, top_p, max_output_tokens, tools=tools, **kwargs
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
            #print(response.candidates[0].grounding_metadata)
            if tools and response.candidates[0].grounding_metadata:
                grounding_data = self._fetch_grounding_data(response)
            return {"response": response.text, "grounding_data": grounding_data}
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")

    def generate_for_gemini_json(self, system_prompt: str, message: str, schema: Type[BaseModel],
                                 temperature: float = 0.0, top_p: float = 0.95, max_output_tokens: int = 8192,
                                 tools: List = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Gemini で JSON を生成する

        Args:
            system_prompt (str): システムプロンプト。
            message (str): ユーザーメッセージ。
            schema (Type[BaseModel]): Pydantic モデルの型。
            temperature (float): ランダム性 (0.0 が最も決定的)。
            top_p (float):  確率が高い上位トークンのセットから選択する確率の閾値。
            max_output_tokens (int): 最大出力トークン数。
             **kwargs: その他の GenerateContentConfig パラメータ (例: safety_settings)。

        Returns:
            Dict[str, Any]: 生成された JSON (辞書形式)。
        """
        grounding_data = None
        contents = self._set_contents(message)

        config = self._create_generation_config(
            system_prompt, temperature, top_p, max_output_tokens, response_schema=schema, response_mime_type="application/json", tools=tools, **kwargs
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
            #print(response)
            if tools and response.candidates[0].grounding_metadata:
                grounding_data = self._fetch_grounding_data(response)
            return {"response": json.loads(response.text), "grounding_data": grounding_data}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating JSON: {e}")

    def generate_chunk_for_gemini(self, system_prompt: str, message: str,
                                  temperature: float = 0.0, top_p: float = 0.95, max_output_tokens: int = 8192,
                                  tools: List = None, **kwargs: Any) -> None:
        """
        Gemini でテキストをストリーミング生成する

        Args:
            system_prompt (str): システムプロンプト。
            message (str): ユーザーメッセージ。
            temperature (float): ランダム性 (0.0 が最も決定的)。
            top_p (float):  確率が高い上位トークンのセットから選択する確率の閾値。
            max_output_tokens (int): 最大出力トークン数。
            **kwargs: その他の GenerateContentConfig パラメータ (例: safety_settings)。
        """
        contents = self._set_contents(message)
        config = self._create_generation_config(
            system_prompt, temperature, top_p, max_output_tokens, tools=tools, **kwargs
        )
        try:
            for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
            ):
                print(chunk.text, end="")
        except Exception as e:
            raise RuntimeError(f"Error generating text stream: {e}")

    def generate_for_gemini_with_input_image(self, system_prompt: str, image_data: str, message: str,
                                              temperature: float = 0.0, top_p: float = 0.95, max_output_tokens: int = 8192,
                                              tools: List = None, **kwargs: Any) -> str:
        """
        Gemini で画像とテキストを組み合わせて生成する

        Args:
            system_prompt (str): システムプロンプト。
            image_data (str): Base64 エンコードされた画像データ (PNG)。
            message (str): ユーザーメッセージ。
            temperature (float): ランダム性 (0.0 が最も決定的)。
            top_p (float):  確率が高い上位トークンのセットから選択する確率の閾値。
            max_output_tokens (int): 最大出力トークン数。
            **kwargs: その他の GenerationConfig パラメータ (例: safety_settings)。

        Returns:
            str: 生成されたテキスト。
        """
        grounding_data = None
        contents = self._set_contents(message, image_data)
        config = self._create_generation_config(
            system_prompt, temperature, top_p, max_output_tokens, tools=tools, **kwargs
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
            if tools and response.candidates[0].grounding_metadata:
                grounding_data = self._fetch_grounding_data(response)
            return {"response": response.text, "grounding_data": grounding_data}
        except Exception as e:
            raise RuntimeError(f"Error generating text with image: {e}")

class Geminijob:
    def __init__(self, state):
        self.gemini = GeminiService()
        self.state  = state
        self.tools = [
            types.Tool(retrieval=types.Retrieval(vertex_ai_search=types.VertexAISearch(datastore=f'projects/{os.getenv("GOOGLE_CLOUD_PROJECT")}/locations/global/collections/default_collection/dataStores/{os.getenv("DATASTORE")}'))),
        ]

    def flow(self) -> State:
        while self.state.retry_count < 3 and not self.state.current_judge:
            self.passive_goal_create()
            self.selection_role()
            self.answer()
            self.check_answer()
        return self.state

    def passive_goal_create(self):
        system_prompt = """
        ユーザーの質問とユーザーの会話履歴を組み合わせて分析し、明確なユーザーのニーズを生成してください。
        ニーズは、質問と会話履歴の内容に基づいて具体的である必要があります。
        要件：
        1. ニーズは明確である必要があります。
        2. 以下の手順に従い回答の生成を行ってください。
            - 会話履歴と質問を組み合わせて分析する。
            - ユーザーのニーズを生成する。

        4. 決して2.以外の行動を取ってはいけません。

        """
        res = self.gemini.generate_for_gemini_json(system_prompt, self.state.query, PassiveGoal.schema())
        self.state.passive_goal = res.get("response").get("user_needs")

    def selection_role(self):
        role_options = "\n".join([f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()])
        system_prompt = f"""要望を分析し、最も適切な回答担当ロールを選択してください。
        選択肢:
        {role_options}
        回答は選択肢の番号（1）のみを返してください。
        """
        res = self.gemini.generate_for_gemini_text(system_prompt, self.state.passive_goal, max_output_tokens=1)
        selected_role = ROLES.get(res.get("response"), ROLES["1"])["name"]
        self.state.current_role = selected_role

    def answer(self):
        role = self.state.current_role
        role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
        system_prompt = f"""
        あなたは{role}として回答してください。あなたの役割に基づいてユーザーのニーズを満たす回答を提供してください。
        役割の詳細に従って回答を生成してください。
        また、回答はLIXILのデータソースにあるもののみ使用してください。
        出典を必ず明記してください
        役割の詳細:
        {role_details}
        """
        res = self.gemini.generate_for_gemini_text(system_prompt, self.state.passive_goal, tools=self.tools)
        self.state.messages.append(res.get("response"))
        self.state.grounding_data = res.get("grounding_data", [])

    def check_answer(self):
        passive_goal = self.state.passive_goal
        answer = self.state.messages[-1] if self.state.messages else ""
        system_prompt = """以下の回答の品質をチェック'False'または'True'を回答してください。
        品質をチェック項目を全て満たしている場合のみ'True'を回答することができます。
            1. 回答の日本語に問題がない
            2. 回答が空文字でない
        また、その判断理由も説明してください。
        """
        check_prompt = f"""
        ユーザーからのニーズ: {passive_goal}
        回答: {answer}
        """
        res = self.gemini.generate_for_gemini_json(system_prompt, check_prompt, Judgement.schema())
        self.state.current_judge = res.get("response").get("judge", False)
        self.state.judgement_reason = res.get("response").get("reason", "")
        if not self.state.current_judge:
            self.state.retry_count += 1