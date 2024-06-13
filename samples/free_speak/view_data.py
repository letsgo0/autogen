"""
该文件用于打印输出
"""
import json


def main():
    # filename = "free_speak-2024-06-04-20-28-05"
    filename = "compare-2024-06-05-15-16-35"

    with open(filename, 'rt', encoding='utf-8') as f:
        result = json.load(f)
        chat_group_history = result.get("chat").get("chat_history")
        review_history = result.get("review").get("chat_history")
        print(f'\n{"+"*5}小组对话{"+"*100}\n')
        for msg in chat_group_history:
            print(f'{"-"*5}role: {msg.get("role", "****")}{"-"*50}')
            print(f'\t{msg.get("content", "****")}')
        print(f'\n{"+" * 5}小组对话摘要{"+" * 100}\n')
        print(f'{result.get("chat").get("summary")}')

        print(f'\n{"+"*5}小组总结{"+"*100}\n')
        for msg in review_history:
            print(f'{"-"*5}role: {msg.get("role", "****")}{"-"*50}')
            print(f'\t{msg.get("content", "****")}')

        print(f'\n{"+"*5}小组总结摘要{"+"*100}\n')
        print(f'{result.get("review").get("summary")}')



if __name__ == "__main__":
    main()
