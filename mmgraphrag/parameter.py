from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

@dataclass
class QueryParam:
    response_type: str = "Keep the responses as brief and accurate as possible. If you need to present information in a list format, use (1), (2), (3), etc., instead of numbered bullets like 1., 2., 3. "
    top_k: int = 10
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_local_context: int = 6000
    # alpha: int = 0.5
    number_of_mmentities: int = 3

cache_path = './cache'

embedding_model_dir = './cache/models/sentence-transformers/all-MiniLM-L6-v2'
EMBED_MODEL = SentenceTransformer(embedding_model_dir, device="cpu")
# EMBED_MODEL = SentenceTransformer(embedding_model_dir, trust_remote_code=True, device="cuda:0")

def encode(content):
    return EMBED_MODEL.encode(content)
"""
def encode(content):
    return EMBED_MODEL.encode(content, prompt_name="s2p_query", convert_to_tensor=True).cpu()
"""

mineru_dir = "./example_input/mineru_result"
API_KEY = "bce-v3/ALTAK-RE6J5TxSujIgjCGj3KxTT/e35f514dfa34f62324061af8730327b29b1166d1"
# MODEL = "moonshot-v1-32k"
MODEL = "deepseek-v3"
# URL = "https://api.moonshot.cn/v1"
# https://qianfan.baidubce.com/v2/chat/completions
# URL = "https://qianfan.baidubce.com/v2/chat/completions"
URL = "https://qianfan.baidubce.com/v2"

MM_API_KEY = "bce-v3/ALTAK-RE6J5TxSujIgjCGj3KxTT/e35f514dfa34f62324061af8730327b29b1166d1"
# MM_MODEL = "qwen-vl-max"
MM_MODEL = "qwen2.5-vl-32b-instruct"
# MM_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MM_URL = "https://qianfan.baidubce.com/v2/chat/completions"
MM_URL = "https://qianfan.baidubce.com/v2"



ZHIPU_API_KEY="2cd149cdfcfb4ea0ae9a25d2b736c6d8.9i0jTAQOAWRg8mr5"
ZHIPU_MODEL="glm-4-plus"

API_KEY = ZHIPU_API_KEY
MODEL = ZHIPU_MODEL
URL = "https://open.bigmodel.cn/api/paas/v4/"


# zhipu多么台
MM_API_KEY = "2cd149cdfcfb4ea0ae9a25d2b736c6d8.9i0jTAQOAWRg8mr5"
# MM_MODEL = "qwen-vl-max"
MM_MODEL = "glm-4v-plus-0111"
# MM_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MM_URL = "https://open.bigmodel.cn/api/paas/v4/"
