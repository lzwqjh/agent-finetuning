def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as f:
        # 逐行读取 JSONL 文件
        lines = f.readlines()

    # 创建一个空的 JSON 字符串
    json_data = ""

    # 解析每一行 JSONL 数据，并将其添加到 JSON 字符串中
    for line in lines:
        # 去除行尾的换行符
        line = line.strip()
        # 在行尾添加逗号和换行符
        line += ",\n"
        # 将行数据添加到 JSON 字符串中
        json_data += line

    # 去除最后一行的逗号和换行符
    json_data = json_data.rstrip(",\n")

    # 将 JSON 字符串写入 JSON 文件
    with open(json_file, 'w') as f:
        f.write("[\n" + json_data + "\n]")

# 指定输入的 JSONL 文件路径和输出的 JSON 文件路径
jsonl_file_path = '/data/ssd/zqh/LLM-RLHF-Tuning/sft_data/openassistant_best_replies_train.jsonl'
json_file_path = '/data/ssd/zqh/LLM-RLHF-Tuning/sft_data/openassistant_best_replies_train.json'

# 调用函数进行转换
jsonl_to_json(jsonl_file_path, json_file_path)