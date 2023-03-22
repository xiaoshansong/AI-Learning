#Yahoo！PlaceFinder API  
#导入urllib  
import urllib  
#导入json模块  
import json  
  
#利用地名，城市获取位置经纬度函数  
def geoGrab(stAddress,city):  
    #获取经纬度网址  
    apiStem='http://where.yahooapis.com/geocode?'  
    #初始化一个字典，存储相关参数  
    params={}  
    #返回类型为json  
    params['flags']='J'  
    #参数appid  
    params['appid']='ppp68N8t'  
    #参数地址位置信息  
    params['location']=('%s %s', %(stAddress,city))  
    #利用urlencode函数将字典转为URL可以传递的字符串格式  
    url_params=urllib.urlencode(params)  
    #组成完整的URL地址api  
    yahooApi=apiStem+url_params  
    #打印该URL地址  
    print('%s',yahooApi)  
    #打开URL，返回json格式的数据  
    c=urllib.urlopen(yahooApi)  
    #返回json解析后的数据字典  
    return json.load(c.read())  
  
from time import sleep  
#具体文本数据批量地址经纬度获取函数  
def massPlaceFind(fileName):  
    #新建一个可写的文本文件，存储地址，城市，经纬度等信息  
    fw=open('places.txt','wb+')  
    #遍历文本的每一行  
    for line in open(fileName).readlines();  
        #去除首尾空格  
        line =line.strip()  
        #按tab键分隔开  
        lineArr=line.split('\t')  
        #利用获取经纬度函数获取该地址经纬度  
        retDict=geoGrab(lineArr[1],lineArr[2])  
        #如果错误编码为0，表示没有错误，获取到相应经纬度  
        if retDict['ResultSet']['Error']==0:  
            #从字典中获取经度  
            lat=float(retDict['ResultSet']['Results'][0]['latitute'])  
            #维度  
            lng=float(retDict['ResultSet']['Results'][0]['longitute'])  
            #打印地名及对应的经纬度信息  
            print('%s\t%f\t%f',%(lineArr[0],lat,lng))  
            #将上面的信息存入新的文件中  
            fw.write('%s\t%f\t%f\n',%(line,lat,lng))  
        #如果错误编码不为0，打印提示信息  
        else:print('error fetching')  
        #为防止频繁调用API，造成请求被封，使函数调用延迟一秒  
        sleep(1)  
    #文本写入关闭  
    fw.close()  
#球面距离计算及簇绘图函数  
def distSLC(vecA,vecB):  
    #sin()和cos()以弧度未输入，将float角度数值转为弧度，即*pi/180  
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)  
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*\  
        cos(pi*(vecB[0,0]-vecA[0,0])/180)  
    return arcos(a+b)*6371.0  
  
import matplotlib  
import matplotlib.pyplot as plt  
  
#@numClust：聚类个数，默认为5  
def clusterClubs(numClust=5):  
    datList=[]  
    #解析文本数据中的每一行中的数据特征值  
    for line in open('places.txt').readlines():  
        lineArr=line.split('\t')  
        datList.append([float(lineArr[4]),float(lineArr[4])])  
        datMat=mat(datList)  
        #利用2-均值聚类算法进行聚类  
        myCentroids,clusterAssing=biKmeans(datMat,numClust,\  
            distMeas=distSLC)  
        #对聚类结果进行绘图  
        fig=plt.figure()  
        rect=[0.1,0.1,0.8,0.8]  
        scatterMarkers=['s','o','^','8'.'p',\  
            'd','v','h','>','<']  
        axprops=dict(xticks=[],ytick=[])  
        ax0=fig.add_axes(rect,label='ax0',**axprops)  
        imgP=plt.imread('Portland.png')  
        ax0.imshow(imgP)  
        ax1=fig.add_axes(rect,label='ax1',frameon=False)  
        for i in range(numClust):  
            ptsInCurrCluster=datMat[nonzero(clusterAssing[:,0].A==i)[0],:]  
            markerStyle=scatterMarkers[i % len(scatterMarkers))]  
            ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\  
                ptsInCurrCluster[:,1].flatten().A[0],\  
                    marker=markerStyle,s=90)  
        ax1.scatter(myCentroids[:,0].flatten().A[0],\  
            myCentroids[:,1].flatten().A[0],marker='+',s=300)  
        #绘制结果显示  
        plt.show()  