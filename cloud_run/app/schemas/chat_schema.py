from typing import Optional
from pydantic import BaseModel, Field
from typing import List, Annotated
import operator

class ChatRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="チャットを行うユーザーの一意の識別子。",
    )
    session_id: str = Field(
        ...,
        description="チャットセッションの一意の識別子。",
    )
    message: str = Field(
        ...,
        description="ユーザーからの入力メッセージ。",
    )

class ChatResponse(BaseModel):
    message: str = Field(
        ...,
        description="AIからの応答メッセージ。",
    )
    grounding_data: Optional[List[dict]] = Field(
        ...,
        description="AIが回答生成に利用した情報ソース",
    )


class State(BaseModel):
    query: str = Field(..., description="ユーザーからの質問")
    user_id: str = Field(
        default="", description="ユーザーの一意の識別子"
    )
    session_id: str = Field(
        default="", description="会話のスレッドID"
    )
    history: List[dict] = Field(
        default=[], description="過去の会話履歴"
    )
    current_role: str = Field(
        default="", description="選定された回答ロール"
    )
    passive_goal: str = Field(
        default="", description="ユーザーからの質問から目的を設計する"
    )
    messages: List[str] = Field(
        default_factory=list, description="回答履歴"
    )
    grounding_data: List[dict] = Field(
        default_factory=list, description="グラウンディングデータ"
    )
    current_judge: bool = Field(
        default=False, description="品質チェックの結果"
    )
    is_hallucination: bool = Field(
        default=False, description="ハルシネーションの有無"
    )
    judgement_reason: str = Field(
        default="", description="品質チェックの判定理由"
    )
    retry_count: int = Field(
        default=0, description="品質チェックのリトライ回数"
    )


class Judgement(BaseModel):
    judge: bool = Field(default=False, description="判定結果")
    reason: str = Field(default="", description="判定理由")


class PassiveGoal(BaseModel):
    user_needs: str = Field(default="", description="ユーザーの要望")


class IdList(BaseModel):
    id_list: list[str] = Field(default=[], description="idのリスト")


class SelectAction(BaseModel):
    apparel_feature: list[str] = Field(default=[], description="ユーザーのニーズから抽出したアパレルに関する特徴")
    suggest_action: bool = Field(default=False, description="提案を行うべきかの判定")
