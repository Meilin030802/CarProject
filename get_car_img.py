import glob

import requests
from lxml import etree
import os

import shutil


def clear_folder(folder_path='./ts'):
    # 使用 shutil.rmtree() 删除文件夹及其内容
    try:
        shutil.rmtree(folder_path)
        # 重新创建空文件夹
        os.makedirs(folder_path)
    except Exception as e:
        print(f"删除文件夹 {folder_path} 失败: {e}")
        raise 'h'


class GetCarImg(object):

    def __init__(self):
        self.counter = 0
        self.headers = {
            "User_Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

    def get_next_url(self):
        self.counter += 1
        return f'https://car-web-api.autohome.com.cn/car/search/searchcar?searchtype=5&pageindex={self.counter}&pagesize=30&orderid=0&state=2'

    def get_id(self):
        car_id_list = []
        while True:
            response = requests.get(self.get_next_url(), headers=self.headers)
            if response.json()['message'] != 'success':
                break
            for i in response.json()['result']['seriesgrouplist']:
                car_id_list.append(i['seriesid'])
        return car_id_list

    def get_img_url(self, url):
        req = requests.get(url, headers=self.headers)
        html = etree.HTML(req.text)
        try:
            img_url_l = html.xpath('//*[@id="data-img-list"]//ul/li[1]/a/img/@data-webp')[0]
            img_url_r = html.xpath('//*[@id="data-img-list"]//ul/li[3]//a/img/@data-webp')[0]
            name1 = html.xpath('//*[@id="data-img-list"]/div/div[1]/h2/a/text()')[0]
        except IndexError:
            pass
        else:
            return ('https:' + img_url_l, name1), ('https:' + img_url_r, name1)
        return []
        # print(req.text)
        # print('https:' + img_url_l, 'https:' + img_url_r, sep='\n')
        # input()

    def download_image(self, url, n, save_folder='./ts'):
        import time
        global file_number
        file_number += 1
        try:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{file_number}.jpg")
            while os.path.exists(save_path):
                save_path = os.path.join(save_folder, f"{file_number}.jpg")
            response = requests.get(url, self.headers)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print(f"{n},图片已保存到 {save_path}")
                name1 = str(file_number) + '.jpg,' + n + '\n'
                return name1
            else:
                print(f"下载失败，状态码：{response.status_code}")
                return False
        except Exception as e:
            print(f"发生错误: {e}")
            return False

    def get_all_url(self, car_id):
        all_url_list = []
        url = f'https://car.autohome.com.cn/pic/series-t/{car_id}.html'
        req = requests.get(url, headers=self.headers)
        html = etree.HTML(req.text)
        for i in html.xpath('//dl[@class="search-pic-cardl"]//a/@href'):
            all_url_list.append('https://car.autohome.com.cn/' + i)
        if len(all_url_list) > 10:
            all_url_list = all_url_list[0:10]
        return all_url_list

    def main(self):
        global name
        car_id = self.get_id()
        for i in car_id:
            for url in self.get_all_url(i):
                na1 = ''
                for k, n in self.get_img_url(url):
                    na = self.download_image(k, n)
                    if not na:
                        clear_folder()  # 清空暂存文件夹
                        break
                    else:
                        na1 += na
                else:
                    img_path_list = glob.glob('./ts/*')
                    for img_path in img_path_list:
                        shutil.move(img_path, f'./img/' + img_path.split("\\")[-1])
                    clear_folder()  # 清空暂存文件夹
                    name += na1
                with open('name.csv', 'w', encoding='utf-8') as f:
                    f.write(name)


file_number = 0


def main():
    clear_folder()
    gci = GetCarImg()
    gci.main()


name = ''
if __name__ == '__main__':
    main()
    print(name)
