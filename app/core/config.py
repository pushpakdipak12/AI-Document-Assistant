from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    APP_NAME: str = "AI Support Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    GROQ_API_KEY: str = ""


settings = Settings()