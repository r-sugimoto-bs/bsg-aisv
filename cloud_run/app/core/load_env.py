import os


def load_environment():
    # ローカル環境の場合のみ.envファイルを読み込む
    if os.path.exists('app/.env'):
        from dotenv import load_dotenv
        load_dotenv('app/.env')