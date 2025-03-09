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


@Benchmark.register(
    name='Finance_instruct',
    dataset_id=os.path.dirname(__file__),
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
            # 构建JSON文件路径
            file_path = os.path.join(dataset_path, "Finance_instruct_test.json")
            
            # 确保文件存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到数据集文件：{file_path}")
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 初始化子集数据
            data_dict[subset] = {}
            
            # 构建测试集数据
            test_data = []
            for item in data:
                
        
                # 构建提示词
                prompt_text = item["instruction"]



                test_item = {
                    'question': prompt_text,
                    'answer': item["output"]
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



    def match(self, gold: str, pred: str, input_d: dict) -> float:
        
        """
        Match the gold answer and the predicted answer.
        """

        llm_judger_sys_prompt = """你是一个金融题目结果评分助手，我会给你一个'参考答案'与一个'模型回复'。你的任务是将模型回复与参考答案进行比较，并判断其正确性，如果回复正确，就输出1，否则输出0。
    # 参考答案: {gold}
    # 模型回复: {pred}
    # 回复要求：按照以上规则给出判断，并在最后将判断结果1 or 0放在boxed{{}}中，例如boxed{{1}} or boxed{{0}}
    """

        messages = [
            {
                "role": "user", 
                "content": llm_judger_sys_prompt.format(gold=gold, pred=pred)
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
                if score > 1:
                    score = 1
                exact_score = score
                return exact_score

        return 0.0  # 如果没有找到匹配项

    
