import argparse
import re
import json
import jsonlines
# from apex.contrib.test.bottleneck.test_bottleneck_module import ground_truth
from datasets import load_from_disk
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from math_verify import parse, verify

from src.models import get_llm_for_sequence_regression
from src.utils import get_tokenizer
from src.utils.logging_utils import init_logger
from multiprocessing import Pool
from transformers import AutoTokenizer

logger = init_logger(__name__)
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def math_equal(gold, answer):
    try:
        gold = parse(gold)
        answer = parse(answer)
        return verify(gold, answer)
    except:
        return False


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                    strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                    + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i: min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


class RuleBasedRMProxy:
    def __init__(self, args):
        self.args = args
        self.prompt2answer = {}
        self.prompt2difficulty = {}
        self.count=0

        dataset = load_from_disk(args.data_path)
        train_list = list(dataset["train"])
        validation_list = list(dataset["test"])

        for line in train_list:
            self.prompt2answer[line['context'].strip()] = str(line['answer'])
            self.prompt2difficulty[line['context'].strip()] = str(line['difficulty'])
        for line in validation_list:
            self.prompt2answer[line['context'].strip()] = str(line['answer'])
            self.prompt2difficulty[line['context'].strip()] = "hard"

        self.timeout_seconds = 2
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.boxed_pattern = re.compile(
            r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")
        # self.boxed_pattern = re.compile(r"\\boxed\{([^{}]*)\}")
        self.valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
        self.repeat_pattern = re.compile(r'(.{10,}?)\1{4,}')  # originally 5/4

    def get_score(self, query):
        try:
            with timeout(self.timeout_seconds):
                if args.template_type == "qwen":
                    prompt = \
                        query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[0].strip()
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
                    if "<|im_end|>" not in response and "<|endoftext|>" not in response:
                        return -1.0
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split(
                        "<|endoftext|>")[0].strip()
                elif args.template_type == "deepseek":
                    prompt = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
                    prompt = prompt.replace("Please reason step by step, and put your final answer within \\boxed{}",
                                            "").strip()
                    response = query.split("<｜Assistant｜>")[-1].strip()

                score = None
                self.count+=1
                ######################
                # 根据prompt, response, prompt2answer等写reward function
                # 思路:
                # 1.从模型生成的response中提取模型的最终答案(hint: 分析sft数据输出最终答案的格式，使用self.boxed_pattern提取答案)
                # 2.用prompt2answer获取prompt对应的groundtruth
                # 3.用math_equal函数评估response是否正确，并据此给出reward
                # 偷到的情报说，答错全部给-1.0，答对给1.0就可
                # 去掉所有的相对路径
                ######################
                
                # base_score = 0.0
                # cot_bonus = 0.0
                # # raise ValueError("{}".format(self.prompt2difficulty[prompt]))
                
                # 1
                model_answer_match = self.boxed_pattern.search(response)
                if not model_answer_match:
                    # No boxed answer found, penalize the score
                    return -1.0
                model_answer = model_answer_match.group(1).strip()

                # # Extract the chain-of-thought (CoT) reasoning text before the boxed answer
                # cot_text = response[:model_answer_match.start()].strip()

                # 2
                true_answer = self.prompt2answer.get(prompt)
                # difficulty = self.prompt2difficulty.get(prompt) # 引入人为bias或许是不对的
                if true_answer is None:
                    # no answer in the dataset
                    return 0.0

                # 3 correctness
                is_correct = math_equal(true_answer, model_answer)
                minus_penalty=max(-0.5,min(0.0,-0.25*((self.count-40000)/40000)))
                score=1.0 if is_correct else 0.0 if self.count<=40000 else minus_penalty
                # 假如在训练120轮后还没有答对,-0.2; 200轮后还没有答对，-0.5.

                # base_scores = {"easy": 1.0, "medium": 1.1, "hard": 1.2}
                # penalty_scores = {"easy": -0.3, "medium": -0.4, "hard": -0.5}
                # if is_correct:
                #     base_score += base_scores.get(difficulty, 1.0)
                # else:
                #     base_score += penalty_scores.get(difficulty, 0.0)

                # # 3.5 thought process
                # # maybe not that important?

                # # key words that can evaluate the quality of the answer

                # elements = {
                #     'transition': (['therefore', ' thus ', 'because', 'since', 'hence'], 0.2),
                #     'rebuttal': (['but', 'however', 'although', 'despite'], 0.3),
                #     'verification': (['verify', 'check', 'confirm', 'think'], 0.2),
                #     'steps': (['step', 'calculate', 'equation', 'reason', 'explain', 'clarify'], 0.3)
                # }

                # for ele in elements.values():
                #     words, bonus = ele
                #     count = sum(cot_text.lower().count(w) for w in words)
                #     cot_bonus += min(bonus * count/3.0, bonus)
                    
                # cot_bonus_weights={'easy': 0.2, 'medium': 0.15, 'hard': 0.1}
                # cot_bonus_weight=0.02 if is_correct else cot_bonus_weights.get(difficulty, 0.1)

                # # 4 penalty
                # # repeat pattern check
                # repeat_penalty = 0.2 if self.repeat_pattern.search(response) else 0.0

                # # useless or meaningless cots
                # length_penalty = 0.3 if len(cot_text) < 100 else 0.0

                # penalty = repeat_penalty + length_penalty
                # penalty_weight = 0.0 if is_correct else 1.0

                # # score = base_score * (1.0-cot_bonus_weight) + cot_bonus * cot_bonus_weight - penalty  # for encouraging reasoning
                # score = base_score*(1-cot_bonus_weight) + cot_bonus*cot_bonus_weight-penalty*penalty_weight # for basic testing
                return score

        except TimeoutException:
            logger.warning("Processing timed out")
            return -1.0
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return -1.0
            # raise e

    def get_reward(self, queries):
        scores = []
        for query in queries:
            score = self.get_score(query)
            scores.append(score)
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rule")
    # RuleBasedRM Parameters
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_gen_len", type=int)
    parser.add_argument("--template_type", type=str, default="qwen", choices=["qwen", "deepseek"])
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)  # for
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    if args.mode == "model":
        reward_model = RewardModelProxy(args)
    else:
        reward_model = RuleBasedRMProxy(args)

    # test_case="<im_start>\nsystem\nnihao<|im_end|>\n<|im_start|>user\n1+1<|im_end|>\n<|im_start|>assistant\n1+1=\\boxed{2}<im_end>"
    # reward=reward_model.get_reward([test_case for _ in range(4)])
    # print(reward)
    # exit()
    app = FastAPI()


    @app.post("/get_reward")
    async def get_reward(request: Request):
        client_host = request.client.host
        logger.info(f"client_ip: {client_host}")
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)


    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
