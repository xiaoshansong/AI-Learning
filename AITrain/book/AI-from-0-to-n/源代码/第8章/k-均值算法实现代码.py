#导入numpy库  
from numpy import *  
#K-均值聚类辅助函数  
  
#文本数据解析函数  
def numpy import *  
    dataMat=[]  
    fr=open(fileName)  
    for line in fr.readlines():  
        curLine=line.strip().split('\t')  
        #将每一行的数据映射成float型  
        fltLine=map(float,curLine)  
        dataMat.append(fltLine)  
    return dataMat  
  
#数据向量计算欧式距离      
def distEclud(vecA,vecB):  
    return sqrt(sum(power(vecA-vecB,2)))  
  
#随机初始化K个质心(质心满足数据边界之内)  
def randCent(dataSet,k):  
    #得到数据样本的维度  
    n=shape(dataSet)[1]  
    #初始化为一个(k,n)的矩阵  
    centroids=mat(zeros((k,n)))  
    #遍历数据集的每一维度  
    for j in range(n):  
        #得到该列数据的最小值  
        minJ=min(dataSet[:,j])  
        #得到该列数据的范围(最大值-最小值)  
        rangeJ=float(max(dataSet[:,j])-minJ)  
        #k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值  
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)  
    #返回初始化得到的k个质心向量  
    return centroids  
  
#k-均值聚类算法  
#@dataSet:聚类数据集  
#@k:用户指定的k个类  
#@distMeas:距离计算方法，默认欧氏距离distEclud()  
#@createCent:获得k个质心的方法，默认随机获取randCent()  
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):  
    #获取数据集样本数  
    m=shape(dataSet)[0]  
    #初始化一个(m,2)的矩阵  
    clusterAssment=mat(zeros((m,2)))  
    #创建初始的k个质心向量  
    centroids=createCent(dataSet,k)  
    #聚类结果是否发生变化的布尔类型  
    clusterChanged=True  
    #只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化  
    while clusterChanged:  
        #聚类结果变化布尔类型置为false  
        clusterChanged=False  
        #遍历数据集每一个样本向量  
        for i in range(m):  
            #初始化最小距离最正无穷；最小距离对应索引为-1  
            minDist=inf;minIndex=-1  
            #循环k个类的质心  
            for j in range(k):  
                #计算数据点到质心的欧氏距离  
                distJI=distMeas(centroids[j,:],dataSet[i,:])  
                #如果距离小于当前最小距离  
                if distJI<minDist:  
                    #当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)  
                    minDist=distJI;minIndex=j  
        #当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法  
        if clusterAssment[i,0] !=minIndex:clusterChanged=True  
        #更新当前变化样本的聚类结果和平方误差  
        clusterAssment[i,:]=minIndex,minDist**2  
    #打印k-均值聚类的质心  
    print centroids  
    #遍历每一个质心  
    for cent in range(k):  
        #将数据集中所有属于当前质心类的样本通过条件过滤筛选出来  
        ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]  
        #计算这些数据的均值（axis=0：求列的均值），作为该类质心向量  
        centroids[cent,:]=mean(ptsInClust,axis=0)  
    #返回k个聚类，聚类结果及误差  
    return centroids,clusterAssment  