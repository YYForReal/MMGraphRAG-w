# -*- coding: utf-8 -*-
import sys
import io
import locale

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
print('starting!d(^_^o)')

import warnings
# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from mmgraphrag import MMGraphRAG
from time import time

pdf_path = './example_input/2020.acl-main.45.pdf'
WORKING_DIR = './example_output'
question = "How does the paper propose to calculate the coefficient \u03b1 for the Weighted Cross Entropy Loss?"

def index():
    rag = MMGraphRAG(
        working_dir=WORKING_DIR,
        input_mode=2
    )
    start = time()
    rag.index(pdf_path)
    print('success!ヾ(✿▽ﾟ)ノ')
    print("indexing time:", time() - start)

def query():
    rag = MMGraphRAG(
        working_dir=WORKING_DIR,
        query_mode = True,
    )
    print(rag.query(question))

# 新增批量测试函数
def batch_test():
    import json
    from pathlib import Path
    
    # 输入输出路径配置
    input_path = Path('./example_input/13_qa_zh.json')
    output_path = Path('./example_output/13_qa_zh_results.json')
    
    # 初始化RAG
    rag = MMGraphRAG(
        working_dir='./example_output',
        query_mode=True
    )
    
    # 加载测试问题
    with open(input_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # 执行批量测试
    results = []
    for case in test_cases:
        answer = rag.query(case['question'])
        results.append({
            'question': case['question'],
            'expected_answer': case['answer'],
            'generated_answer': answer
        })
    
    # 保存测试结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'测试完成，结果已保存至 {output_path}')

if __name__ == "__main__":
    # index()
    # query()
    batch_test()