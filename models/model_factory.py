from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from .models import GPT2, GPT_NEO, OpenELM, PHI2
from .language_model import LanguageModel


class ModelFactory:
    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> LanguageModel:
        if "gpt2" == model_args.architecture:
            return GPT2(model_args=model_args, env_args=env_args)
        elif "gptneo" == model_args.architecture:
            return GPT_NEO(model_args=model_args, env_args=env_args)
        elif "openelm" == model_args.architecture:
            return OpenELM(model_args=model_args, env_args=env_args)
        elif "phi2" == model_args.architecture:
            return PHI2(model_args=model_args, env_args=env_args)
        else:
            raise ValueError(model_args.architecture)
