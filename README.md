## 大数据应用分类标注挑战赛深度学习模型Baseline

**比赛链接：**
- http://challenge.xfyun.cn/2019/gamedetail?type=detail/classifyApp

## 使用方式
- 下载好数据存放`/data`文件夹下，创建`data/xfyun`文件夹
- 数据预处理生成数据集`python xunfei/pre_classify.py`
- 训练模型`python xunfei_classify.py -d ./data/xfyun -n xunfei -m train`
- 预测模型`python xunfei_classify.py -d ./data/xfyun -n xunfei -m predict`

## keras模型训练方式
> 使用bert提取特征，Bidirectional 训练模型,可以达到 74.39565。需要一定的内存。	
```
pip install keras
pip install keras_bert

# 指定数据目录，
data_dir = './data/'
# 将bert预训练数据放到改目录
BERT_PRETRAINED_DIR = './chinese_L-12_H-768_A-12/'
# 运行
python3 xf_bert_v2.py

```

## 交流方式
![微信：wangmouren7400](img/wechatid.jpg)

## 参考资料
- [tokenizer](https://www.cnblogs.com/bymo/p/9675654.html)
- [模型结构](https://blog.csdn.net/asialee_bird/article/details/88813385)

@**Galen**_20190717_