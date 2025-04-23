import os
import json
import jsonlines
import pandas 
import datasets
import argparse
import datasets
import pandas as pd
import random
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer


from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("/mnt/data/Qwen2.5-Math-1.5B")

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except Exception:
        return False

def to_messages(line):
    messages=[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": line["prompt"]},
    ]
    text=line["prompt"]
    return text, messages, line['answer']

def save_dataset(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(dataset, Dataset):
        dataset.save_to_disk(save_dir)
    elif isinstance(dataset, DatasetDict):
        dataset.save_to_disk(save_dir)
    else:
        raise ValueError("")

def main(args):
    data=[]
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            data.append(line)
        
    print(len(data))
    save_data=[]
    # easy_data=[]
    # medium_data=[]
    # hard_data=[]
    for line in data:
        text, messages, answer=to_messages(line)
        difficulty=line["meta"]["difficulty"]
        s={"dataset": "rl", "context": text, "context_messages": messages, "answer": str(answer),"difficulty":str(difficulty), "source": "math.3k"}
        if len(tokenizer.encode(text))>=310: continue
        if not is_jsonable(s):
            print(f"Non-JSON-serializable data: {s}")
            raise ValueError("Data is not JSON-serializable")
        save_data.append(s)
    #     if difficulty=="easy":
    #         easy_data.append(s)
    #     elif difficulty=="medium":
    #         medium_data.append(s)
    #     else:
    #         hard_data.append(s)
    # # 尝试采用分层混合，不需要随机，需要用到每一条数据，分为5部分
    # # 1. easy的2/3+medium的1/3+hard的1/6
    # # 2. easy剩下的1/3+medium接下来的1/3+hard的1/3
    # # 3. medium剩下的1/3+hard剩下的1/2
    # easy_split1 = int(len(easy_data) * 2/3)
    
    # medium_split1 = int(len(medium_data) * 1/3)
    # medium_split2 = int(len(medium_data) * 2/3)  # 累计3/4处
    
    # hard_split1 = int(len(hard_data) * 1/6)
    # hard_split2 = int(len(hard_data) * 1/2)  # 累计1/2处
    
    # part_1=easy_data[:easy_split1] + medium_data[:medium_split1] + hard_data[:hard_split1]
    # part_2=easy_data[easy_split1:] + medium_data[medium_split1:medium_split2] + hard_data[hard_split1:hard_split2]
    # part_3=medium_data[medium_split2:] + hard_data[hard_split2:]
    
    # random.shuffle(part_1)
    # _part_1=part_1.copy()
    # random.shuffle(_part_1)
    # save_data+=part_1+_part_1
    # random.shuffle(part_2)
    # _part_2=part_2.copy()
    # random.shuffle(_part_2)
    # save_data+=part_2+_part_2
    # random.shuffle(part_3)
    # _part_3=part_3.copy()
    # random.shuffle(_part_3)
    # save_data+=part_3+_part_3
    
    evaldata=[]
    with open("/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/data/eval/RL.jsonl", 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            evaldata.append(line)
    eval_save_data=[]
    for line in evaldata:
        text, messages, answer=to_messages(line)
        if len(tokenizer.encode(text))>=310: continue
        s={"dataset": "rl", "context": text, "context_messages": messages, "answer": str(answer).strip(),"source": line['source']}
        if not is_jsonable(s):
            print(f"Non-JSON-serializable data: {s}")
            raise ValueError("Data is not JSON-serializable")
        eval_save_data.append(s)
    
    training_data=save_data
    valiation_data=eval_save_data
    
    print(json.dumps(training_data[-1], indent=4))    
    print(json.dumps(valiation_data[-1], indent=4))
    training_data=pd.DataFrame(training_data)
    validation_data=pd.DataFrame(valiation_data)
    
    print("training_data_length", len(training_data))
    print("validation_data_length", len(validation_data))

    data_dict={
        "train": training_data,
        "test": validation_data,
    }
    dataset_dict=DatasetDict({
        split_name: Dataset.from_pandas(df) for split_name,df in data_dict.items()
    })
    save_dataset(dataset_dict, args.output_path)



if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/data/train/math3k_rl_prompt.jsonl")
    parser.add_argument("--output_path", type=str, default="/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/data/train/math3k_rl_prompt")
    args=parser.parse_args()
    main(args)
