## 大数据应用分类标注挑战赛深度学习模型Baseline

**比赛链接：**
- http://challenge.xfyun.cn/2019/gamedetail?type=detail/classifyApp

## 使用方式
- 下载好数据存放`/data`文件夹下，创建`data/xfyun`文件夹
- 数据预处理生成数据集`python xunfei/pre_classify.py`
- 训练模型`python xunfei_classify.py -d ./data/xfyun -n xunfei -m train`
- 预测模型`python xunfei_classify.py -d ./data/xfyun -n xunfei -m predict`

## 交流QQ群
- 群号：826192597

## 参考资料
- [tokenizer](https://www.cnblogs.com/bymo/p/9675654.html)
- [模型结构](https://blog.csdn.net/asialee_bird/article/details/88813385)

@**Galen**_20190717_