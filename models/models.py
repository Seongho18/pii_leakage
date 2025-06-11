from transformers import GPT2Config, GPTNeoConfig

from .language_model import LanguageModel


class GPT2(LanguageModel):
    """ A custom convenience wrapper around huggingface gpt-2 utils """

    def get_config(self):
        return GPT2Config()


class GPT_NEO(LanguageModel):
    """ A custom convenience wrapper around huggingface gpt-neo utils """

    def get_config(self):
        return GPTNeoConfig()


class OpenELM(LanguageModel):
    """ A custom convenience wrapper around huggingface openelm utils """


class PHI2(LanguageModel):
    """ A custom convenience wrapper around huggingface llama utils """

