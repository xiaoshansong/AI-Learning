# 风格迁移

## 运行

--首先打开命令行

输入如下命令即可执行
"python neural_style.py --content <content file> --styles <style file> --output <output file>"

同时还需下载预训练的imagenet-vgg-verydeep-19.mat文件
"python neural_style.py --network /vgg/imagenet-vgg-verydeep-19.mat --content <content file> --styles <style file> --output <output file>"

使用 `--iterations` 改变迭代参数，程序默认为1000次。
"python neural_style.py --network /vgg/imagenet-vgg-verydeep-19.mat --content <content file> --styles <style file> --output <output file> --iterations 1000"


对于3968*2976 的原图图, 采用 GTX 1060 GPU，进行1000次迭代总共花费 120 秒。

## 系统环境依赖需求
numpy
Pillow
scipy
tensorflow-gpu >= 1.0