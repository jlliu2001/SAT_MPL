import os
import openai

# 你可以将API KEY通过参数传入，或用环境变量OPENAI_API_KEY

def expand_labels_with_gpt(natural_language, api_key=None, model="gpt-3.5-turbo"):
    """
    使用OpenAI GPT API将自然语言描述转为label名称列表。
    返回: list of label names
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env variable.")
    openai.api_key = api_key

    prompt = f"""
你是一个医学图像分割专家。请根据以下描述，列出所有需要分割的脑区label名称，输出一个Python列表，列表元素为每个label的中文名称，不要有多余解释。
描述：{natural_language}
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    # 解析返回的label列表
    content = response["choices"][0]["message"]["content"]
    try:
        label_list = eval(content)
        if not isinstance(label_list, list):
            raise ValueError
    except Exception:
        raise ValueError(f"GPT返回内容无法解析为列表: {content}")
    return label_list 