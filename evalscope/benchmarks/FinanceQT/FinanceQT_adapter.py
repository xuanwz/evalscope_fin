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
    name='FinanceQT',
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
            file_path = os.path.join(dataset_path, "FinanceQT.json")
            
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

        llm_judger_sys_prompt = """我会给你一个有关交易策略的'代码问题',除此之外还有一个'参考回答'和一个'待打分回答'，请根据'参考回答'给'待打分回答'一个评分。
#优先先判定'待打分回答'是不是代码，如果不是，直接评分为0；如果是，再参考下面评分标准。
#评分标准：
    1、如果待打分回答的代码和参考回答一致，或能正常运行且解决问题，评分为1；
    2、如果待打分回答的代码不能解决问题或不能正常运行，评分为0；
    3、如果待打分回答的代码出现严重逻辑错误，评分为0.
    现在请你根据下列信息打分：
#代码问题：{question}
#参考回答：{gold}
#待打分回答：{pred}

# 回复要求：只给出评分，将分数放在boxed{{}}中。严禁输出思考过程，你的输出只能是回复示例中的其中一种。
# 回复示例：boxed{{0}}，boxed{{1}}
    """
        
        messages = [
            {
                "role": "user", 
                "content": llm_judger_sys_prompt.format(question=input_d['question'], gold=gold, pred=pred)
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

    
