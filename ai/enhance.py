import os
import json
import sys

import dotenv
import argparse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from structure import Structure
if os.path.exists('.env'):
    print('Load .env', file=sys.stderr)
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek/deepseek-chat-v3-0324:free')
    openai_base_url = os.environ.get("OPENAI_BASE_URL", 'https://openrouter.ai/api/v1')
    language = os.environ.get("LANGUAGE", 'Chinese')

    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data

    print('Open:', args.data, file=sys.stderr)

    llm = ChatOpenAI(model=model_name,
                     base_url=openai_base_url)
    
    # 创建一个包含结构信息的JSON提示模板
    json_template = """
    {system}

    {template}

    请以JSON格式回答，必须包含以下字段：
    {{
    "tldr": "简短摘要",
    "motivation": "研究动机",
    "method": "使用方法",
    "result": "研究结果",
    "conclusion": "结论"
    }}

    语言: {language}
    内容: {content}
    """
    
    parser = JsonOutputParser(pydantic_object=Structure)
    
    print('Connect to:', model_name, file=sys.stderr)
    prompt = PromptTemplate(
        template=json_template,
        input_variables=["content", "language"],
        partial_variables={"system": system, "template": template}
    )

    chain = prompt | llm | parser

    for idx, d in enumerate(data):
        try:
            response = chain.invoke({
                "language": language,
                "content": d['summary']
            })
            d['AI'] = response
            print(response)
        except langchain_core.exceptions.OutputParserException as e:
            print(f"{d['id']} has an error: {e}", file=sys.stderr)
            d['AI'] = {
                 "tldr": "Error",
                 "motivation": "Error",
                 "method": "Error",
                 "result": "Error",
                 "conclusion": "Error"
            }
        with open(args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl'), "a") as f:
            f.write(json.dumps(d) + "\n")

        print(f"Finished {idx+1}/{len(data)}", file=sys.stderr)

if __name__ == "__main__":
    main()
