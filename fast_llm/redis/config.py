from fast_llm.config import Config, Field, FieldHint, config_class


@config_class()
class RedisConfig(Config):
    host: str = Field(
        default="localhost",
        desc="Hostname or IP address of the Redis server.",
        hint=FieldHint.core,
    )

    port: int = Field(
        default=6379,
        desc="Port number on which the Redis server is running.",
        hint=FieldHint.core,
    )

    stream_key: str = Field(
        default=None,
        desc="Name of the Redis stream to read data from.",
        hint=FieldHint.core,
    )
