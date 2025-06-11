from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

import dp_transformers
from huggingface_hub import login

import numpy as np
import torch
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    TrainerCallback

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset


@dataclass
class GeneratedText:
    text: str  # the generated text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])

class dpDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.input_ids = []
        self.attention_mask = []
        for sample in dataset:
            if len(sample['input_ids']) == 0:
                print(sample)
                continue
            input_id = torch.tensor(sample['input_ids'])
            self.input_ids.append(input_id)
            self.attention_mask.append(torch.ones_like(input_id))

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

    def __len__(self):
        return len(self.input_ids)

class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """ A wrapper class around a huggingface LM.
        """
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """ Gets the maximum size of the context """
        if "gpt-neo" in self.model_args.architecture:
            return self._lm.config.max_position_embeddings
        else:
            return self._lm.config.n_positions

    @abstractmethod
    def tokenizer(self):
        """ Returns this model's tokenizer. """
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError


    def load_gpt(self, architecture: str, verbose: bool) -> 'LanguageModel':
        # load model
        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    f"> Loading the provided {architecture} checkpoint from "
                    f"'{self.model_args.model_ckpt}'."
                )
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_ckpt, return_dict=True).eval()
        else:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {architecture} model.")
            self._lm = AutoModelForCausalLM.from_pretrained(
                architecture, return_dict=True, trust_remote_code=True).eval()

        # load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            architecture#, use_fast=self.model_args.tokenizer_use_fast
        )
        num_added_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        mean_tok_emb = self._lm.transformer.wte.weight.data.mean(dim=0)
        self._lm.resize_token_embeddings(len(self._tokenizer))
        # Initialize the newly-added token embedding to the mean of all token embeddings
        for i in range(num_added_toks):
            self._lm.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
        return

    def load_openelm(self, verbose: bool) -> 'LanguageModel':
        login(token='Model_Access_Token')
        # load model
        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    "> Loading the provided apple/OpenELM-270M checkpoint "
                    f"from '{self.model_args.model_ckpt}'."
                )
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_ckpt,
                return_dict=True,
                trust_remote_code=True
            ).eval()
        else: # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(
                    "> Loading a public, pre-trained apple/OpenELM-270M model."
                )
            self._lm = AutoModelForCausalLM.from_pretrained(
                'apple/OpenELM-270M', return_dict=True, trust_remote_code=True
            ).eval()
        # load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            trust_remote_code=True,
            model_max_length=4096
        )
        self._lm.config.use_cache = False
        num_added_toks = self._tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'}
        )
        self._lm.resize_token_embeddings(len(self._tokenizer))
        return


    def load_phi2(self, verbose: bool) -> 'LanguageModel':
        # laod model
        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    "> Loading the provided microsoft/phi-2 "
                    f"checkpoint from '{self.model_args.model_ckpt}'."
                )
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_ckpt,
                return_dict=True,
                trust_remote_code=True,
                torch_dtype="auto"
            ).eval()
        else:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(
                    "> Loading a public, pre-trained microsoft/phi-2"
                    " model."
                )
            self._lm = AutoModelForCausalLM.from_pretrained(
                'microsoft/phi-2',
                return_dict=True,
                trust_remote_code=True,
                torch_dtype="auto"
            ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/phi-2',
            trust_remote_code=True,
            model_max_length=4096
        )
        self._lm.config.use_cache = False
        num_added_toks = self._tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'}
        )
        self._lm.resize_token_embeddings(len(self._tokenizer))
        return


    def load(self, verbose: bool = False) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint.
        """

        if self.model_args.architecture == 'gpt2':
            self.load_gpt('gpt2', verbose)
        elif self.model_args.architecture == 'gptneo':
            self.load_gpt('EleutherAI/gpt-neo-125m', verbose)
        elif self.model_args.architecture == 'openelm':
            self.load_openelm(verbose)
        elif self.model_args.architecture == 'llama3':
            self.load_llama3(verbose)
        elif self.model_args.architecture == 'llama3.2':
            self.load_llama3_2(verbose)
        elif self.model_args.architecture == 'phi2':
            self.load_phi2(verbose)
        else:
            raise ValueError(self.model_args.architecture)
        self._lm.generation_config.pad_token_id = self._tokenizer.pad_token_id

        self._lm.to(self.env_args.device)
        return self

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, sampling_args) -> List[GeneratedText]:
        """ Helper function to generate a single batch of text.
        """
        self._lm.eval()

        input_len = input_ids.size(1)
        out = self._lm.generate(
            input_ids=input_ids.to(self.env_args.device),
            attention_mask=attention_mask.to(self.env_args.device),
            max_length=input_len + sampling_args.seq_len,
            do_sample=sampling_args.do_sample,
            temperature=sampling_args.temp,
            top_k=sampling_args.top_k,
            top_p=sampling_args.top_p,
            output_scores=False,
            return_dict_in_generate=True
        )

        generated_texts = []
        for text in self._tokenizer.batch_decode(out.sequences, skip_special_tokens=False):
            generated_texts.append(text)
        return generated_texts

    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using the sampling args.
        """
        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] * r if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt] * r
        )

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        generated_data: List[GeneratedText] = []
        num_batches = int(np.ceil(sampling_args.N / self.env_args.eval_batch_size))
        for _ in tqdm(
                range(num_batches),
                disable=not sampling_args.generate_verbose,
                desc="Generating with LM"
        ):
            generated_data.extend(self.generate_batch(input_ids, attention_mask, sampling_args))
        return generated_data[:sampling_args.N]

    def tokenize_datasets(self, datasets: List[RealDataset], column_name="text") -> List:
        """ Tokenizes the 'text' column of a list of dataset using this model's tokenizer """
        tokenize_function = lambda x: self._tokenizer(x[column_name], truncation=True)
        return [dataset.get_hf_dataset().map(tokenize_function, batched=True).select_columns(['input_ids', 'attention_mask']) for dataset in datasets]

    def perplexity(self, data: Union[list, str], offset=0, max_length=0, apply_exp=True, verbose=True,
                   return_as_list: bool = False) -> float:
        """ Compute the perplexity of the model on a string.
        """
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of tokens viewed
        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
            input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
            target_ids = input_ids.clone()

            if offset > 0:  # ignore everything up to the offset
                target_ids[:, :offset] = -100

            tgt_len = (target_ids.size(1) - offset)
            if max_length > 0:  # ignore everything except offset:offset+max_length
                target_ids[:, offset + max_length:] = -100
                tgt_len = max_length

            with torch.no_grad():
                outputs = self._lm(input_ids, labels=target_ids)
            loss, logits = outputs[:2]
            if torch.isnan(loss):
                continue
            if return_as_list:
                nlls.append(loss.cpu().detach())
            else:
                nlls.append(loss.cpu().detach())
                ctr += tgt_len
        self._lm.training = original_mode
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())

    def _fine_tune_dp(self,
                      train_dataset: RealDataset,
                      eval_dataset: RealDataset,
                      train_args: TrainerArgs,
                      privacy_args: PrivacyArgs):

        with train_args.main_process_first(desc="Tokenizing datasets"):
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])

        self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
                                                            target_epsilon=privacy_args.target_epsilon,
                                                            target_delta=privacy_args.target_delta,
                                                            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
            tokenizer=self._tokenizer
        )

        # Workaround for modern `transformers` which removed `use_cuda_amp` 
        # (See https://github.com/huggingface/transformers/pull/25702)
        trainer.use_cuda_amp = False

        try:
            trainer.train()
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })

        trainer.save_model()
        self._lm.eval()

    def fine_tune(self,
                  train_dataset,
                  eval_dataset,
                  train_args: TrainerArgs,
                  privacy_args: PrivacyArgs):
        """ Fine-Tune the LM with/without DP
        """

        if privacy_args.target_epsilon > 0:
            return self._fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
        return self._fine_tune(train_dataset, eval_dataset, train_args)

    def _fine_tune(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        """
        if extra_callbacks is None:
            extra_callbacks = []

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Train and Eval Datasets ..")
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"

        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)

        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

        trainer.save_model()
        self._lm.eval()
