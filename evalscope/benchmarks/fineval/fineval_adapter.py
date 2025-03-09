from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import exact_match
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.utils import ResponseParser
import os

@Benchmark.register(
    name='fineval',
    dataset_id=os.path.dirname(__file__),
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['accounting', 'finance', 'economy','certificate'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt='',  # noqa: E501
)
class IQuizAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D']

    def load_from_disk(self, dataset_path: str, subset_list: list = None, work_dir: str = None, **kwargs) -> dict:
        """
        从本地磁盘加载数据集
        
        Args:
            dataset_path: 数据集文件夹路径
            subset_list: 子集列表 ['accounting', 'finance', 'economy', 'certificate']
            work_dir: 工作目录
            
        Returns:
            dict: 格式为 {'subset_name': {'split_name': data}}
        """
        import pandas as pd
        import os
        
        # 初始化数据字典
        data_dict = {}
        
        # 如果没有提供subset_list，使用类中定义的默认子集列表
        if subset_list is None:
            subset_list = ['accounting', 'finance', 'economy', 'certificate']
        
        # 遍历每个子集
        for subset in subset_list:
            # 构建Excel文件路径
            file_path = os.path.join(dataset_path, f"fineval_{subset}.xlsx")
            
            # 确保文件存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到数据集文件：{file_path}")
            
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 初始化子集数据
            data_dict[subset] = {}
            
            # 构建测试集数据
            test_data = []
            prompt='你是一个金融知识专家，下面是一道中国金融相关考试的问题，请选出其中的正确答案。你可以一步步思考，并在最后将最终答案的选项放入 \\boxed{}',
            for _, row in df.iterrows():
                item = {
                    'question': f'{prompt}\n{row["question"]}',
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
        return ResponseParser.parse_first_option_with_choices(result, self.choices)

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        return exact_match(gold=gold, pred=pred)
    
