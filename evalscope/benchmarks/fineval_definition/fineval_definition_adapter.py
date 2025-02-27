from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import exact_match
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.utils import ResponseParser
import requests
import time
import re


@Benchmark.register(
    name='fineval_definition',
    dataset_id=r'D:\working_projects\compass\evalscope_fin\evalscope\benchmarks\fineval_definition',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['main'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt='请回答给出的金融名词定义问题，不要有除了答案外其余的回复。',  # noqa: E501
)
class IQuizAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.choices = []
        self.eval_need_q = True


    def load_from_disk(self, dataset_path: str, subset_list: list = None, work_dir: str = None, **kwargs) -> dict:

        import pandas as pd
        import os
        
        # 初始化数据字典
        data_dict = {}
        
        # 如果没有提供subset_list，使用类中定义的默认子集列表
        if subset_list is None:
            subset_list = ['main']
        
        # 遍历每个子集
        for subset in subset_list:
            # 构建Excel文件路径
            file_path = os.path.join(dataset_path, "fineval_definition.xlsx")
            
            # 确保文件存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到数据集文件：{file_path}")
            
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 初始化子集数据
            data_dict[subset] = {}
            
            # 构建测试集数据
            test_data = []
            for _, row in df.iterrows():
                item = {
                    'question': row['question'],
                    'answer': row['answer']
                }
                test_data.append(item)
            
            # 将测试集数据添加到对应的子集中
            data_dict[subset][self.eval_split] = test_data
        
        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:

        prompt = f"问题: {input_d['question']}\n"
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
        llm_judger_sys_prompt = """我会给你一个'题目'，一个'标准回答'与一个'待打分回答'，请根据评分标准对'待打分回答'进行打分。
    ## 评分标准：
    # 1. 如果回答准确，与正确回答大体上含义一致，加2分
    # 2. 如果回答完整，覆盖了正确回答的所有关键点，加1分
    # 3. 如果回答高效，无太多冗余信息，加1分
    # 4. 如果回答丰富，具有正确回答之外一些额外的正确信息，加1分
    # 5. 满分共5分，你只有0，1，2，3，4, 5六个选项
    # 回复要求：按照标准给出评分理由，并在最后将分数放在boxed{}中
    """
        llm_judger_user_prompt = f"题目: {input_d['question']}\n标准回答: {gold}\n待打分回答: {pred}"
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
            r'分数[:：]\s*(\d+)',
            r'得分[:：]\s*(\d+)', 
            r'(\d+)\s*分'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                exact_score = score/5
                return exact_score

        return 0.0  # 如果没有找到匹配项

    
