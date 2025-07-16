from setuptools import setup, find_packages

setup(
    name="lixil_aisv_app",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        # 必要なら requirements.txt を読み込んでリスト化してもOK
    ],
)