from google.cloud import firestore
from typing import List,Dict
import os

class FireStore:
    def __init__(self):
        self.client = firestore.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'), database=os.getenv('FIRESTORE_DATABASE'))


    #全てのセッションの会話履歴を取得
    def user_chat_log(self, user_id: str) -> dict:
        doc_ref = self.client.collection('history').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return {}


    #特定のセッションの会話履歴を取得
    def fetch_chat_log_to_input(self, user_id: str, session_id: str) -> list:
        doc_ref = self.client.collection('history').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            chat_data = doc.to_dict().get(session_id)
        else:
            chat_data = []
        return chat_data


    #特定のセッションに会話内容を追加
    def insert_current_chat(self, user_id: str, session_id: str, user_message: str, agent_message: str) -> None:
        doc_ref = self.client.collection('history').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            chat_data = doc.to_dict()
            target_data = chat_data.get(session_id, [])
            target_data.append({'user-message': user_message, 'agent-message': agent_message})

            doc_ref.update({session_id: target_data})
        else:
            # ドキュメントが存在しない場合は新規作成
            doc_ref.set({session_id: [{'user-message': user_message, 'agent-message': agent_message}]})

        print("会話履歴を更新しました")

    def delete_chat_log(self, user_id: str, session_id: str) -> None:
        # ドキュメントのパス
        doc_ref = self.client.collection('history').document(user_id)
        doc_ref.update({session_id: firestore.DELETE_FIELD})