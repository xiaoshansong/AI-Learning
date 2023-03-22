import tkinter
from PIL import Image,ImageDraw
from Inference import inference

class MyCanvas:
    """
    设置一个256*256大小的容器进行手写界面的绘制
    背景色设置为黑色，绘制轨迹设置为白色
    """
    def __init__(self,root):
        self.root=root
        self.canvas=tkinter.Canvas(root,width=256,height=256,bg='black')
        self.canvas.pack()
        self.image1 = Image.new("RGB", (256, 256), "black")
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>',self.Draw)

    # 绘制轨迹
    def Draw(self,event):
        self.canvas.create_oval(event.x,event.y,event.x,event.y,outline="white",width = 20)
        self.draw.ellipse((event.x-10,event.y-10,event.x+10,event.y+10),fill=(255,255,255))


def main():
    # 建立一个tkinter对象,设置大小为380*300
    root = tkinter.Tk()
    root.geometry('380x300')
    # 创建一个256*256的框架容纳手写的容器，位于tkinter对象的左边，填充y方向
    frame = tkinter.Frame(root, width=256, height=256)
    frame.pack_propagate(0)
    frame.pack(side="left", fill='y')
    # 将frame导入canvas容器
    canvas1 = MyCanvas(frame)
    # 创建一个图像识别的实例
    infer = inference()

    # 定义识别按钮触发函数
    # 按下的时候将cavas导出为图片，放入infer中进行图像识别，并将结果显示在label2中
    def inference_click():
        img = canvas1.image1
        result = infer.predict(img)
        result = int(result)
        label2["text"] = str(result)

    # 定义清除按钮的触发函数
    # 按下的时候将canvas情况并重新绘制背景，并将label设置为空
    def clear_click():
        canvas1.canvas.delete("all")
        canvas1.image1 = Image.new("RGB", (256, 256), "black")
        canvas1.draw = ImageDraw.Draw(canvas1.image1)
        label2["text"] = ""

    # 定义识别按钮的样式
    botton_Inference = tkinter.Button(root,
                                      text="检测",
                                      width=14,
                                      height=2,
                                      command=inference_click
                                      )
    # 定义清除按钮的样式
    botton_Clear = tkinter.Button(root,
                                      text="清屏",
                                      width=14,
                                      height=2,
                                      command=clear_click
                                      )
    # 绑定识别按钮到tkinter中，设置位置为顶层
    botton_Inference.pack(side="top")

    # 绑定清除按钮到tkinter中
    botton_Clear.pack(side="top")

    # 定义label1
    label1 = tkinter.Label(root, justify="center", text="检测结果为:")
    label1.pack(side="top")

    # 定义label2
    label2 = tkinter.Label(root, justify="center")

    # 设置字体样式与大小
    label2["font"] = ("Arial, 48")
    label2.pack(side="top")
    root.mainloop()

if __name__ == '__main__':
    main()
