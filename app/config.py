from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Stories GPT RAG"
    openai_api_key: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
