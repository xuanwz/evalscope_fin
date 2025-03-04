from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='gpt-4o',   # 模型名称 (需要与部署时或者调用api的模型名称一致)
    api_url='https://www.apillm.online/v1/chat/completions',  # 推理服务地址
    api_key='sk-nsIxq2HnRLDlg7g8C699A6B6CbD045CdB1F0DcBd4811Bb37',
    api_type='request', #这里根据你API的方式选择request还是openai方式，默认是openai
    eval_type='service',   # 评测类型，SERVICE表示评测推理服务
    datasets=[
    # 'math_500',  # 数据集名称
    'fineval', 'fineval_definition','math_500'   
    ],
    dataset_args={ #EvalScope内置支持，每个数据集的具体配置。请查看每个数据集的adapter文件，选择它的subset。
                   #也就是数据集的子集。不填默认该数据集全测。
    'fineval': {'subset_list': ['finance','accounting'], 'few_shot_num': 0},
    'fineval_definition': {'subset_list': ['main'], 'few_shot_num': 0},
    },    
    limit=100, #每个数据集测试的数量，不填默认全测。
    eval_batch_size=5, #测试数据的线程数量，建议不要过大。
    generation_config={       # 模型推理配置
        'max_tokens': 4096,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=False               # 是否使用流式请求，推理类模型可以设置为True防止请求超时
)

run_task(task_cfg=task_cfg)