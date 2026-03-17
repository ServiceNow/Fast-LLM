from fast_llm.config import Config, config_class


@config_class()
class DataConfig(Config):
    _abstract = True
