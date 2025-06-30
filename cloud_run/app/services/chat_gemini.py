from google import genai
from google.genai import types
import base64
from pydantic import BaseModel
import os
import json
from app.schemas.chat_schema import State, PassiveGoal, Judgement

#API化
class GeminiService():
    def __init__(self):
        #TODO パラメータをGitにあげるときに環境変数に置く
        self.project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.location   = 'us-central1'
        self.model      = "gemini-2.0-flash-001"
        self.client     = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        self.safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
                )]


    #config設定
    def setting_generation_config(self, system_prompt, temperature = 0, top_p = 0.95, max_output_tokens = 8192, response_modalities = ["TEXT"], response_mime_type = None, response_schema = None):
        return types.GenerateContentConfig(
                temperature         =  temperature,
                top_p               = top_p,
                max_output_tokens   = max_output_tokens,
                response_modalities = response_modalities,    #今後のアップデートで画像とテキスト、音声出力をする際に変更する可能性あり
                safety_settings     = self.safety_settings,
                system_instruction  = [types.Part.from_text(text = system_prompt)],
                response_mime_type  = response_mime_type,
                response_schema     = response_schema
        )


    #contetntsをセットする関数
    def set_contents(self, message, image_data=None):
        contents = []
        if image_data:
                #base64形式のデータをデコードし、定義してある形式に代入
                #任意の画像形式の変換や設定はこちらで行ってください。
                contents.append(types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type="image/png",
                ))
        contents.append(types.Content(
                    role="user",
                    parts=[
                            types.Part.from_text(text = message)
                    ]
                ))
        return contents


    def generate_for_gemini_text(self, system_prompt, message) -> str:

        '''
        contents = ["メッセージ"]でも可
        contentsの内部にモデルの回答を埋め込むことでチャットも可
        '''
        #contentsを定義
        contents = self.set_contents(message)

        #contents = [message]

        response = self.client.models.generate_content(model=self.model, contents=contents,  config = self.setting_generation_config(system_prompt)).text
        return response


    def generate_for_gemini_json(self, system_prompt, message, schema: type[BaseModel]) -> str:

        '''
        contents = ["メッセージ"]でも可
        contentsの内部にモデルの回答を埋め込むことでチャットも可
        '''
        #contentsを定義
        contents = self.set_contents(message)

        #contents = [message]

        response = self.client.models.generate_content(
                    model=self.model, contents=contents,
                    config = self.setting_generation_config(
                                system_prompt,
                                response_mime_type="application/json",
                                response_schema=schema
                                )
                    ).text
        return json.load(response)


    def generate_chunk_for_gemini(self, system_prompt, message) -> str:

        #contentsを定義
        contents = self.set_contents(message)

        for chunk in self.client.models.generate_content_stream(
                model    = self.model,
                contents = contents,
                config   = self.setting_generation_config(system_prompt),
                ):
                print(chunk.text, end="")


    def generate_for_gemini_with_input_image(self, system_prompt, image_data, message):

        '''
        image1には画像の加工処理コードをset_contents関数に追加してください。
        例ではbase64形式をデコードしています
        '''

        #contentsを定義
        contents = self.set_contents(message, image_data)
        response = self.client.models.generate_content(model=self.model, contents=contents, config = self.setting_generation_config(system_prompt)).text
        return response
      
class Geminijob:
    def __init__(self, state):
        self.gemini = GeminiService()
        self.state  = state
    def flow(self) -> State:
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
        self.state.passive_goal = res.get("user_needs")

    def selection_role(self):
        
        return self.gemini.generate_chunk_for_gemini(system_prompt, message)
    
    def answer(self):
    
    def check_answer(self):
        return self.gemini.generate_for_gemini_text()