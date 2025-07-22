# app/modules/firestore.py

from google.cloud import firestore
from typing import List, Dict, Any
import os
from app.schemas.chat_schema import State  # Stateクラスをインポート

class FireStore:
    def __init__(self):
        self.client = firestore.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'), database=os.getenv('FIRESTORE_DATABASE'))
        self.collection_name = 'history' # 既存のコレクション名

    def fetch_chat_log_to_input(self, user_id: str, session_id: str) -> list:
        """セッションの全ての会話履歴（メッセージペアのリスト）を取得"""
        if not user_id or not session_id:
            return []
        doc_ref = self.client.collection(self.collection_name).document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get(session_id, [])
        return []

    def fetch_latest_state_from_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """最新の会話ログから、保存されたState情報のみを読み込む"""
        chat_data = self.fetch_chat_log_to_input(user_id, session_id)
        if not chat_data:
            return {}
        return chat_data[-1].get('state', {})

    def insert_chat_with_state(self, state: State) -> None:
        """現在の会話ペアと、その時点のState情報をマージして保存"""
        if not state.user_id or not state.session_id:
            print("Error: user_id and session_id are required.")
            return

        user_id = state.user_id
        session_id = state.session_id
        
        # 保存するState情報を辞書として準備
        state_to_save = {
            "store_name": state.store_name,
            "branch_name": state.branch_name,
            "passive_goal": state.passive_goal,
            "current_role": state.current_role,
            "asking_slot": state.asking_slot,
        }
        
        # 新しい会話ログのエントリーを作成
        new_log_entry = {
            'user-message': state.query,
            'agent-message': state.messages[-1] if state.messages else "",
            'state': state_to_save
        }

        doc_ref = self.client.collection(self.collection_name).document(user_id)
        doc = doc_ref.get()
        
        if doc.exists:
            all_sessions = doc.to_dict()
            session_history = all_sessions.get(session_id, [])
            session_history.append(new_log_entry)
            doc_ref.update({session_id: session_history})
        else:
            doc_ref.set({session_id: [new_log_entry]})
            
        print(f"✅ Chat and State for session '{session_id}' updated in 'history' collection.")