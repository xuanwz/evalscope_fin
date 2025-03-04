from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import exact_match
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.utils import ResponseParser
import requests
import time
import re
import json
import os
import pyarrow.parquet as pq
import pandas as pd

@Benchmark.register(
    name='senti_data',
    dataset_id=r'D:\working_projects\compass\evalscope_fin\evalscope\benchmarks\senti_data',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['main'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt='',  # noqa: E501
)
class IQuizAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.choices = []
        self.eval_need_q = True


    
    def load_from_disk(self, dataset_path: str, subset_list: list = None, work_dir: str = None, **kwargs) -> dict:
        # 初始化数据字典
        data_dict = {}

        # 如果没有提供subset_list，使用类中定义的默认子集列表
        if subset_list is None:
            subset_list = ['main']

        # 遍历每个子集
        for subset in subset_list:
            # 构建Parquet文件路径
            file_path = os.path.join(dataset_path, "sent_valid.csv")

            # 确保文件存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到数据集文件：{file_path}")

            df = pd.read_csv(file_path)
    
            # 构建提示词模板
            prompt_template = '''Below is a finance-related tweet. Please analyze its sentiment.
            "0" represents "Bearish"
            "1" represents "Bullish"
            "2" represents "Neutral"

            Requirements:
            -Your output should be one of the above labels.
            -Add a "\\n" at the beginning of each output before generating data
            put the final answer in \\boxed{{}}
            Your reasoning should not be less than 2048 tokens.
            
            Text: {text}'''
            # 初始化子集数据
            data_dict[subset] = {}

            # 构建测试集数据
            test_data = []
            for index, row in df.iterrows():
                # 构建question，由instruction、input、question组成


                test_item = {
                    'question': prompt_template.format(text=row['text']),
                    'answer': str(row['label'])
                }
                test_data.append(test_item)

            # 将测试集数据添加到对应的子集中
            data_dict[subset][self.eval_split] = test_data

        return data_dict
    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:

        prompt = f"{input_d['question']}\n"
        return {'data': [prompt], 'multi_choices': self.choices, 'system_prompt': self.system_prompt}

    def __form_options(self, options: list):
        option_str = '选项:\n'
        for opt, choice in zip(options, self.choices):
            option_str += f'({choice}): {opt}' + '\n'
        return option_str

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        return result



    def call_judger(self,messages, retry_count=0):
        retry_limit = 5
        """调用API"""
        try:
            data = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 4096,
            "stream": False,
            "temperature": 0.0,
            "top_p": 0.9
            }
            response = requests.post(
                "https://www.apillm.online/v1/chat/completions",
                headers={
            'Authorization': 'Bearer sk-nsIxq2HnRLDlg7g8C699A6B6CbD045CdB1F0DcBd4811Bb37',
            'Content-Type': 'application/json'
        },
                json=data,
                timeout=400
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f'请求失败，状态码: {response.status_code}, 错误内容: {response.text}')
                if retry_count < retry_limit:
                    print(f'正在重试第 {retry_count + 1} 次...')
                    time.sleep(1)
                    return self.call_judger(messages, retry_count + 1)
                return False
                
        except requests.RequestException as e:
            print(f'API 请求异常: {e}')
            if retry_count < retry_limit:
                print(f'正在重试第 {retry_count + 1} 次...')
                time.sleep(1)
                return self.call_judger(messages, retry_count + 1)
            return "false"



    def match(self, gold: str, pred: str, input_d: dict) -> float:
        
        """
        Match the gold answer and the predicted answer.
        """
        llm_judger_sys_prompt = """我会给你一个'标准回答'与一个'模型回答'，请判断'模型回答'是否与'标准回答'的含义一致。注意，这是一道情感分析题目，答案中的数字"0" represents "Bearish"；"1" represents "Bullish"；"2" represents "Neutral"；如果一致，输出1，否则输出0。
    # 回复要求：按照以上标准给出判断理由，并在最后将判断结果放在boxed{}中，例如boxed{1} or boxed{0}
    """
        llm_judger_user_prompt = f"标准回答: {gold}\n模型回答: {pred}"
        messages = [
            {
                "role": "system",
                "content": llm_judger_sys_prompt
            },
            {
                "role": "user", 
                "content": llm_judger_user_prompt
            }
        ]
        # 调用模型
        response = self.call_judger(messages)
        score_patterns = [
            r'boxed\{(\d+)\}',
            r'判断[:：]\s*(\d+)',
            r'结果[:：]\s*(\d+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                exact_score = score
                return exact_score

        return 0.0  # 如果没有找到匹配项

    
