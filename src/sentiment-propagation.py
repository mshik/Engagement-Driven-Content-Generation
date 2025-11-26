#!/usr/bin/env python
# coding: utf-8
import pickle
from collections import defaultdict
from statistics import harmonic_mean
import pathlib

# In[1]:


hugging_token = "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"

# In[2]:


from huggingface_hub import login

login(hugging_token)

# In[3]:


import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import networkx as nx

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline,
    BitsAndBytesConfig
)
import pandas as pd
import numpy as np
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer#, set_seed
from transformers import set_seed
#from trl.core import LengthSampler
class LengthSampler:
    def __init__(self, min_value: int, max_value: int):
        self.min = int(min_value)
        self.max = int(max_value)

    def __call__(self) -> int:
        if self.max <= self.min:
            return self.min
        # torch.randint upper bound is exclusive
        return torch.randint(self.min, self.max + 1, (1,)).item()

import matplotlib.pyplot as plt

# In[4]:

import readability

from transformers import set_seed

from argparse import ArgumentParser

from utilities import *

set_seed(42)

# In[5]:

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="google/gemma-2b-it",
                                      metadata={"help": "the model name"})  # "NousResearch/Llama-2-7b-chat-hf"
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="lvwerra/distilbert-imdb",
                                             metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    #batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


# parser = HfArgumentParser(ScriptArguments)
# script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
parser = HfArgumentParser((ScriptArguments,))
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
set_seed(script_args.seed)

# In[6]:


# Propagation Model

import os, sys

sys.path.insert(0, "SocialAIGym/src")

from data_component import DataComponent
from information_diffusion_component import BoundedConfidenceDiffusionComponent
from opinion_diffusion_component import FJDiffusionComponent
#from brexit_classifier import StanceClassifier
from graph_utils import *


def main(args):
    TOPIC = args["TOPIC"]
    TYPE = args["TYPE"]
    PROPAGATION = args["PROPAGATION"]
    NETWORK_OPINION = args["NETWORK_OPINION"]
    MODEL_NAME = args["MODEL_NAME"]
    LLM_pos = args["LLM_pos"]
    MODULARITY = args["MODULARITY"]
    HOMOPHILY = args["HOMOPHILY"]

    # synthetic data generator params
    print("*******")
    print(f"TOPIC: {TOPIC}")
    print(f"TYPE: {TYPE}")
    print(f"PROPAGATION: {PROPAGATION}")
    print(f"MODEL NAME: {MODEL_NAME}")
    print(f"LLM POSITION: {LLM_pos}")
    print("*******")

    # bounded confidence model params
    epsilon = 0.2
    mu = 0.5

    if "brexit" in TOPIC or "referendum" in TOPIC:
        print(f"Using REAL NETWORK of {TOPIC}")

        data, position_dict = get_real_data(TOPIC)

        s = TOPIC.split("-")[1]  # positive or negative
        llm_node_id = position_dict[f"{s}-{LLM_pos}"]

    else:
        print(f"NETWORK OPINION: {NETWORK_OPINION}")
        print(f"MODULARITY: {MODULARITY}")
        print(f"HOMOPHILY: {HOMOPHILY}")

        num_nodes = 100
        # modularity = 0.5
        # homophily = 0.5
        # avg_deg = 13

        avg_deg = 5

        alpha, beta, modularity, homophily = get_network_parameters(NETWORK_OPINION=NETWORK_OPINION,
                                                                    HOMOPHILY=HOMOPHILY, MODULARITY=MODULARITY)

        data = DataComponent(num_nodes, modularity, homophily, avg_deg, alpha=alpha, beta=beta)

        if LLM_pos == "echo-low":
            llm_node_id = LLM_in_echochamber(data, "low")
        elif LLM_pos == "echo-high":
            llm_node_id = LLM_in_echochamber(data, "high")
        elif LLM_pos == "comm-largest":
            llm_node_id = LLM_in_comm(data, "largest")
        elif LLM_pos == "comm-smallest":
            llm_node_id = LLM_in_comm(data, "smallest")
        elif LLM_pos == "central":
            llm_node_id = LLM_central(data)

    print(f"LLM NODE ID: {llm_node_id}")

    if PROPAGATION == "bcm":
        data.pre_compute_neighboring()  # save neighbors for each node
        information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data, epsilon=epsilon, mu=mu)
        # llm_node_id = 0

    elif PROPAGATION == "fj":
        information_diffusion_model = FJDiffusionComponent(data_component=data)
        betweenness_centrality = nx.betweenness_centrality(data.get_graph())
        sorted_by_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        # llm_node_id = sorted_by_betweenness[0][0]

        # compute minimum
        zoomed_fj = np.linspace(-5, 5, 500)

        zoomed_results = []

        for l in tqdm(zoomed_fj, total=len(zoomed_fj)):
            zoomed_results.append(
                information_diffusion_model.polarization_plus_disagreement_at_equilibrium(l, llm_node_id))

        fj_minimum = round(min(zoomed_results))

    opinions = information_diffusion_model.get_opinions()

    path = "../results/"  # TO CHANGE

    if "brexit" in TOPIC or "referendum" in TOPIC:  # real data
        config_path = f"initial-config-LLM_{LLM_pos}"
    else:
        config_path = f"initial-config-LLM_{LLM_pos}-MODULARITY_{MODULARITY}-HOMOPHILY_{HOMOPHILY}-NETWORK_OPINION_{NETWORK_OPINION}"

    saving_fn = f"{MODEL_NAME}-{TYPE}-sentiment-propagation-readability-{PROPAGATION}-{TOPIC}"

    utility_saving_path = os.path.join(path, config_path, saving_fn)

    if not os.path.exists(utility_saving_path):
        os.makedirs(utility_saving_path)

    print(
        f'Opinions stats \nmean: {opinions.mean()}\nstd: {opinions.std()}\nmin: {opinions.min()}\nmax: {opinions.max()}')

    _ = plt.hist(opinions, bins=25, range=[0, 1])
    plt.xlabel('opinion value')
    plt.ylabel('occurrences')
    #plt.show()
    plt.savefig(os.path.join(utility_saving_path, f'opinion_distribution.png'))

    # In[8]:

    MODEL_NAMES_DICT = {"gemma2": "google/gemma-2b-it", "gpt2": "gpt2-medium",
                        "llama2": "NousResearch/Llama-2-7b-chat-hf", "mistral": "mistralai/Mistral-7B-v0.1"}
    model_name = MODEL_NAMES_DICT[MODEL_NAME]

    '''
    config = PPOConfig(
        early_stopping=True,
        target_kl=50,
        # init_kl_coef=1,
        model_name=model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed
    )
    '''
    config = PPOConfig(
        # training / logging
        learning_rate=script_args.learning_rate,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        batch_size=script_args.batch_size,             # or local_batch_size/micro_batch_size if you prefer
        mini_batch_size=script_args.mini_batch_size,   # or local_mini_batch_size
        num_ppo_epochs=script_args.ppo_epochs,         # ← was ppo_epochs
        seed=script_args.seed,
        report_to=script_args.log_with,                             # ← replaces log_with
        project=model_name,                             # optional; shows in W&B “Project”

        # PPO / KL controls in 0.25.1
        kl_coef=0.2,                                   # strength of KL penalty
        kl_estimator="kl",                             # "kl" | "js" | "mse" (depends on your preference)

        # (optional) run shaping present in 0.25.1
        total_episodes=None#,                           # or an int if you want to cap episodes
        #response_length=script_args.get("response_length", 128),
        #temperature=script_args.get("temperature", 1.0),
    )

    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.
    def build_dataset(
            tokenizer
    ):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """

        # train_dataset = load_dataset(dataset_name, split="train")

        if TYPE == "completion":
            prompt = f"{TOPIC.capitalize()} are the most"

            if "brexit" in TOPIC:
                prompt = f"Brexit is the most"
            if "referendum" in TOPIC:
                prompt = "2016 Italian constitutional referendum is the most"
                # prompt = f"The impact of {TOPIC.capitalize()} on UK is"

        elif TYPE == "generation":
            prompt = f"Generate a post about {TOPIC}."

        size = config.batch_size
        df = pd.DataFrame(np.repeat(prompt, size), columns=["0"])
        ds = Dataset.from_dict(df)

        train_dataset = ds.rename_columns({"0": "question"})

        original_columns = train_dataset.column_names
        num_proc = 24

        def preprocess_function(examples):
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            for question in examples["question"]:
                #  query = "Question: " + question + "\n\nAnswer: \n"
                query = question
                tokenized_question = tokenizer(query, truncation=True, max_length=script_args.max_length)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

            return new_examples

        ds = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length, batched=False)

        ds.set_format(type="torch")
        return ds

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    #if "decapoda" in config.model_name.lower():
    if "decapoda" in config.project.lower():
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        # required for llama
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        #tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.project)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer)

    print(dataset["query"][0])

    #if "gemma" in config.model_name:
    if "gemma" in config.project:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

    # if "Llama" in config.model_name:
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        #config.model_name,
        config.project,
        load_in_8bit=False,
        device_map="auto",
        #load_in_4bit=True,
        #device_map='cuda:0',
        #device_map=None,
        peft_config=lora_config
    )
    #from accelerate import disk_offload
    #disk_offload(model=model, offload_dir="offload")
    #model.to("cuda")

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )

    # In[14]:

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer
    )

    def generate_response(question, model, tokenizer):
        inputs = tokenizer(question, return_tensors="pt").to(device)

        # correct_inputs = tokenizer(correct_answer, return_tensors="pt").to(inputs_device)

        outputs = model.generate(**inputs, max_new_tokens=150)  # =correct_inputs["input_ids"].shape[1])
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # In[17]:

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

    TRAIN = True

    print(f"TRAIN: {TRAIN}")

    output_min_length = 32
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    if TYPE == "completion":
        N_EPOCHS = 80
        if ("brexit" in TOPIC or "referendum" in TOPIC) and "negative" in TOPIC:
            N_EPOCHS = 500
    elif TYPE == "generation":
        N_EPOCHS = 500

    print(f"ITERATIONS: {N_EPOCHS}")

    # if TOPIC == "brexit":  # real data
    #     basedir = os.path.join(sys.path[0], "..", "data/processed")
    #     pathClf = os.path.join(basedir, f'calibrated_classifier_{"DEBERTA"}_{"95perc"}.pkl')
    #     pathPCA = os.path.join(basedir, f'pca_projector_{"DEBERTA"}_{"95perc"}.pkl')
    #
    #     # Example usage
    #     classifier_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     sentiment_model = StanceClassifier(pathPCA=pathPCA, pathClf=pathClf, device=classifier_device)
    #
    # else:

    sentiment_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

    if "brexit" in TOPIC or "referendum" in TOPIC:  # real data
        # "LLM_pos", "Optimal_reward"
        pass
        # optimal_reward = optimal_rewards[optimal_rewards["LLM_pos"] == LLM_pos]["Optimal_reward"].values[0]

    else:
        optimal_rewards = get_optimal_rewards(TOPIC=TOPIC, create=False)

        # "LLM_pos", "Modularity", "Homophily", "Network_opinion", "Optimal_reward"
        optimal_reward = optimal_rewards[(optimal_rewards["LLM_pos"] == LLM_pos) &
                                         (optimal_rewards["Modularity"] == MODULARITY) &
                                         (optimal_rewards["Homophily"] == HOMOPHILY) &
                                         (optimal_rewards["Network_opinion"] == NETWORK_OPINION)][
            "Optimal_reward"].values[0]
        print(f"OPTIMAL REWARD: {optimal_reward}")

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000
    }

    if TRAIN:

        kl_values = []
        generated_texts = []
        sentiment_scores = []
        total_rewards = []
        propagation_rewards = []
        readability_scores = []

        # We then build the sentiment analysis pipeline, passing the model name and the
        # sentiment analysis pipeline arguments. Let's also make sure to set the device
        # to the same device as the PPOTrainer.

        # reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

        # We then define the arguments to pass to the `generate` function. These arguments
        # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
        # the `generate` function of the trained model.

        try:

            for _ in tqdm(range(N_EPOCHS), total=N_EPOCHS):
                for batch in tqdm(ppo_trainer.dataloader, total=len(ppo_trainer.dataloader)):
                    question_tensors = batch["input_ids"]

                    response_tensors = ppo_trainer.generate(
                        question_tensors,
                        return_prompt=False,
                        length_sampler=output_length_sampler,
                        **generation_kwargs,
                    )
                    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                    print(batch["response"])

                    # Compute sentiment score
                    texts = [q + r for q, r in zip(batch["query"], batch["response"])]

                    generated_texts.append(texts)
                    readabilities = [readability.getmeasures(x, lang='en')['readability grades']["Kincaid"]
                                     for x in texts]

                    readability_scores.append(readabilities)

                    sentiment_outputs = sentiment_model(texts)
                    messages_values = [output["score"] if output["label"] == "POSITIVE" else 1 - output["score"]
                                       for output in sentiment_outputs]

                    sentiment_scores.extend(messages_values)

                    rewards = []

                    for i, message_value in enumerate(messages_values):
                        if PROPAGATION == "bcm":

                            opinion_shift_tot, num_activated_users, _, _ = information_diffusion_model.propagate_message(
                                message=message_value,
                                node_id=llm_node_id)

                            reward = num_activated_users

                        elif PROPAGATION == "fj":
                            pol_dis = information_diffusion_model.polarization_plus_disagreement_at_equilibrium(
                                message_value,
                                llm_node_id)

                            reward = -(pol_dis - fj_minimum) ** 2

                        propagation_rewards.append(reward)

                        # GEOMETRIC MEAN
                        reward = np.sqrt(reward * max(readabilities[i], 0))

                        total_rewards.append(reward)

                        rewards.append(torch.tensor(reward, dtype=torch.float))

                    # Run PPO step
                    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

                    stats["env/readability"] = np.mean(readability_scores[-1])
                    stats["env/sentiment"] = np.mean(messages_values)
                    stats["env/propagation"] = np.mean(propagation_rewards[-len(batch):])

                    ppo_trainer.log_stats(stats, batch, rewards)

                    kls = (stats["objective/logprobs"] - stats["objective/ref_logprobs"]).mean(axis=1)
                    kl_values.append(kls)

                kl = stats["objective/kl"]
                early_stop = ppo_trainer._early_stop(kl)

                if early_stop:
                    print(f"Early stopping... KL is equal {kl}")
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        saving_path = os.path.join(path, config_path, saving_fn, "ppo_checkpoints")

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        model.save_pretrained(saving_path)
        tokenizer.save_pretrained(saving_path)

        kls_path = os.path.join(utility_saving_path, "kl_values.npy")
        texts_path = os.path.join(utility_saving_path, "generated_texts.npy")
        sentiment_path = os.path.join(utility_saving_path, "sentiment_scores.npy")
        readability_path = os.path.join(utility_saving_path, "readability_scores.npy")
        propagation_path = os.path.join(utility_saving_path, "propagation_rewards.npy")
        rewards_path = os.path.join(utility_saving_path, "total_rewards.npy")

        np.save(kls_path, kl_values)
        np.save(texts_path, generated_texts)
        np.save(sentiment_path, sentiment_scores)
        np.save(readability_path, readability_scores)
        np.save(propagation_path, propagation_rewards)
        np.save(rewards_path, total_rewards)

        print("TRAINING RESULTS SAVED.")

    VANILLA_RESULTS = False

    if VANILLA_RESULTS:

        vanilla_total_rewards = []
        vanilla_propagation_rewards = []

        for batch in tqdm(ppo_trainer.dataloader, total=len(ppo_trainer.dataloader)):
            question_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute sentiment score
            vanilla_generated_texts = [q + r for q, r in zip(batch["query"], batch["response"])]

            # generated_texts.append(texts)
            vanilla_readability_scores = [readability.getmeasures(x, lang='en')['readability grades']["Kincaid"]
                                          for x in vanilla_generated_texts]

            sentiment_outputs = sentiment_model(vanilla_generated_texts)
            vanilla_sentiment_scores = [output["score"] if output["label"] == "POSITIVE" else 1 - output["score"]
                                        for output in sentiment_outputs]

            for i, message_value in enumerate(vanilla_sentiment_scores):
                if PROPAGATION == "bcm":
                    opinion_shift_tot, num_activated_users, _, _ = information_diffusion_model.propagate_message(
                        message=message_value,
                        node_id=llm_node_id)

                    reward = num_activated_users

                    vanilla_propagation_rewards.append(reward)

                    vanilla_total_rewards.append(np.sqrt(reward * max(vanilla_readability_scores[i], 0)))

        vanilla_texts_path = os.path.join(utility_saving_path, "vanilla_generated_texts.npy")
        vanilla_sentiment_path = os.path.join(utility_saving_path, "vanilla_sentiment_scores.npy")
        vanilla_readability_path = os.path.join(utility_saving_path, "vanilla_readability_scores.npy")
        vanilla_propagation_path = os.path.join(utility_saving_path, "vanilla_propagation_rewards.npy")
        vanilla_rewards_path = os.path.join(utility_saving_path, "vanilla_total_rewards.npy")

        np.save(vanilla_texts_path, vanilla_generated_texts)
        np.save(vanilla_sentiment_path, vanilla_sentiment_scores)
        np.save(vanilla_readability_path, vanilla_readability_scores)
        np.save(vanilla_propagation_path, vanilla_propagation_rewards)
        np.save(vanilla_rewards_path, vanilla_total_rewards)

        print("VANILLA RESULTS SAVED.")



if __name__ == '__main__':
    TYPE = "completion"  # "generation"  #
    PROPAGATION = "bcm"  # "fj" #
    MODEL_NAME = "gemma2"  # "mistral"  # "llama2"  # "gpt2"  #

    # NETWORK_OPINION = "uniform"  # "positive"  # "negative"  #  "neutral"  #
    # LLM_pos = "central"
    # MODULARITY = "high"
    # HOMOPHILY = "high"
    # TOPIC = "cats"

    parser = ArgumentParser()
    # parser.add_argument("-T", "--TYPE")
    # parser.add_argument("-P", "--PROPAGATION")
    # parser.add_argument("-N", "--MODEL_NAME")
    # parser.add_argument("-C", "--TOPIC")

    parser.add_argument("-T", "--TOPIC")
    parser.add_argument("-L", "--LLM_pos")

    parser.add_argument("-O", "--NETWORK_OPINION", default=None)
    parser.add_argument("-M", "--MODULARITY", default=None)
    parser.add_argument("-H", "--HOMOPHILY", default=None)

    args = vars(parser.parse_args())

    args["TYPE"] = TYPE
    args["PROPAGATION"] = PROPAGATION
    args["MODEL_NAME"] = MODEL_NAME

    # args["TOPIC"] = TOPIC
    # args["NETWORK_OPINION"] = NETWORK_OPINION
    # args["LLM_pos"] = LLM_pos
    # args["MODULARITY"] = MODULARITY
    # args["HOMOPHILY"] = HOMOPHILY

    main(args)
