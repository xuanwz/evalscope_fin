from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType



model_list = ['qwen2.5-32b-instruct','qwen2.5-14b-instruct','qwen2.5-7b-instruct']

for model in model_list:
    task_cfg = TaskConfig(
        model=model,   # 模型名称 (需要与部署时的模型名称一致)
        api_url='https://www.apillm.online/v1/chat/completions',  # 推理服务地址
        api_key='sk-nsIxq2HnRLDlg7g8C699A6B6CbD045CdB1F0DcBd4811Bb37',
        eval_type='service',   # 评测类型，SERVICE表示评测推理服务
        api_type='request',
        #数据集配置
        datasets=[
            'Ant_Finance',
            'Finance_instruct',
            'FinanceIQ',
            'FinanceQT',
            'FinCorpus',
            #'FinCUGE', #数据集质量不高
        ],  
        limit=100, #每个数据集测试的数量
        eval_batch_size=20,   # 测评batch size
        review_batch_size=10, # llm as judge batch size
        
        generation_config={       # 模型推理配置
            'max_tokens': 4096,  # 最大生成token数，建议设置为较大值避免输出截断
            'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
            'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
            'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
        },
        stream=True               # 是否使用流式请求，推荐设置为True防止请求超时
)

    run_task(task_cfg=task_cfg)