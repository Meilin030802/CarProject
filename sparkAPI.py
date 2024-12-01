from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import csv

last = ''
last_a = ''
if __name__ == "__main__":
    # 示例调用

    appid = "54fc7991"
    api_secret = "M2ZkYmQyMGUyYzIzMWM1YzU2NDBmY2Ix"
    api_key = "f92c76cb1f1efbbd9328d0249c839cf9"
    Spark_url = "wss://spark-api.xf-yun.com/v4.0/chat"
    domain = "4.0Ultra"
    spark = ChatSparkLLM(
        spark_api_url=Spark_url,
        spark_app_id=appid,
        spark_api_key=api_key,
        spark_api_secret=api_secret,
        spark_llm_domain=domain,
        streaming=False,
    )

    with open('name_out.csv', mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        first_column = []
        for row in csv_reader:
            first_column.append(row[0])  # 获取每一行的第一列
        # print(first_column)

    with open('name_out.csv', mode='a', encoding='utf-8', newline='') as file_out:
        csv_writer = csv.writer(file_out)
        with open('name_out1.csv', mode='r', encoding='utf-8', newline='') as file:
            csv_reader = csv.reader(file)
            # 逐行读取并打印
            for row in csv_reader:
                a = [row[0], row[1]]
                # print(a[0])  # 图片文件名
                # print(a[1])  # 车型
                if a[0] in first_column:
                    print(a[0] + '已完成跳过')
                    continue
                messages = [ChatMessage(
                    role="user",
                    content=f"""
                            请分析以下四个方面：  
                           1. 汽车类型: 是轿跑、SUV、轿车、卡车、客车？必须从这四个选一个当作答案。
                           2. 外观风格: 是运动还是奢华?必须从这两个选一个当作答案。
                           3. 车身高度: 是高还是矮？必须从这两个选一个当作答案。
                           4. 车轮尺寸: 是大还是小?  必须从这两个选一个当作答案。

                           请按照格式 '[汽车类型]-[外观风格]-[车身高度]-[车轮尺寸]' 输出答案。  
                           例如：“SUV-奢华-高-大”。只有 "SUV-奢华-高-大"，严格按照这个来，不要有多余的字，错了也行  不要解释原因。。

                           汽车型号: {a[1]}"""
                )]

                if a[1] == last:
                    a1 = last_a
                else:
                    while True:
                        handler = ChunkPrintHandler()
                        a1 = spark.generate([messages], callbacks=[handler])
                        if len(a1.generations[0][0].text) < 15 and a1.generations[0][0].text.count('-') == 3:
                            break
                        print('错误' + a1.generations[0][0].text)
                print(a[0], a1.generations[0][0].text)
                st = a1.generations[0][0].text.split("-")
                a.append(st[0])  # 添加属性1
                a.append(st[1])  # 添加属性2
                a.append(st[2])  # 添加属性3
                a.append(st[3])  # 添加属性4
                csv_writer.writerow(a)
                file_out.flush()
                last_a = a1
                last = a[1]
