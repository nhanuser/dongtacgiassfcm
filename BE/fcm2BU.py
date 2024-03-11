import pandas as pd
import numpy as np
from random import random
import random


class FCM1():
    
    
    def __init__(self, *args, **kwargs):
        self.dict_cluster = {}
        self.c = 0
        self.X = np.zeros(1)
        
        return
        
    def read_data(self, data,label):
        self.data = data
        self.data_table = np.array(self.data)
        self.label_temp =label
    
    def preprocess_data(self):
        dt=self.data
        self.value= dt
        self.label = self.label_temp
        self.label_list = pd.unique(self.label)
        self.label_list = self.label_list.tolist()
        self.num_class = 2
        self.label_data =self.label.values.tolist()
        self.label_count = np.array(self.label.value_counts())
        self.X = np.array(self.value)
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        
        self.final_data = self.value
        self.final_data[''] = self.label
        self.final_data_table = np.array(self.final_data)
        
    
    def set_c(self, input):
        self.c = input
    
    def generate_U_ngang(self, k1): #k1: Số lượng phần tử được chọn làm phần tử được giám sát từ bộ dữ liệu ban đầu
        
        self.U_ngang = np.zeros((self.n,self.c))
    
    
        self.index_x_giamsat = random.sample(range(self.n),k1)# CHọn k1 vector x từ n vector x ban đầu  
        index_x_giamsat = np.sort(self.index_x_giamsat)

        for i in range(k1):
            if len(self.dict_cluster) == 0 :
                c1= random.randint(0,self.c-1)
            else:
                c1= self.dict_cluster[self.label_data[index_x_giamsat[i]]]
                #Chọn vị trí cụm giám sát
            self.U_ngang[index_x_giamsat[i]][c1] = 0.5     
        self.sum_i= sum(self.U_ngang.T)
        
    def generate_V(self):
        self.V = np.random.rand(self.c,self.p) 
        
    # Tinh d_ki (trong cong thuc 6)
    def d_ki(self,k, i):
        kq = sum(  pow(self.X[k] - self.V[i], 2) )
        return kq     
    
        #Cong thuc so 6



    def cong_thuc_6(self, m): #Don't clear
        Z = np.zeros((self.n,self.c))
        #tu day suy ra U = U_ngang + Z ta tinh Z

        D = np.zeros((self.n,self.c))
        for k in range(self.n):
            for i in range(self.c):
                D[k][i] = pow(self.d_ki(k,i),1/(1-m))    
            D[k] = D[k]/sum(D[k])    
      
        #Z = (1/d_ki)^1/m-1z = np.array (i = 1 ,c)
        for i in range(self.n) :
            Z[i] = (1 - self.sum_i[i] )*D[i]
        
        return self.U_ngang+Z  # tuc la U  
    
    def cong_thuc_4(self,m): #Tính tâm của cụm
        V_temp = np.zeros((self.c,self.p))
        U_temp = abs(self.U - self.U_ngang)
        U_temp = pow(U_temp,m)
        for i in range(self.c):
            tu_so = np.zeros(self.p)
            for k in range(self.n):
                tu_so +=U_temp[k][i] * self.X[k] 
            V_temp[i] = tu_so/ sum((U_temp.T)[i])
        
        return V_temp
    
    # đếm số lượng phần tử mỗi cụm tương ứng với từng lớp nhãn dữ liệu
    def count_class(self, num_class, num_cluster):
        self.count_class_cluster = np.zeros((num_class,num_cluster), dtype="int64")
        self.cluster=np.zeros((self.n))
        for k in range(self.n):
            k_class = self.label_list.index(self.label_data[k])
            index_max = np.argmax(self.U[k])
            self.cluster[k] =index_max
            self.count_class_cluster[k_class][index_max] +=1 
        # print(self.count_class_cluster)     
        # Ma trận có num_class dòng và num_cluster cột, phần tử thứ i,j của ma trận thuộc class i, cluster j
        
     # Các độ đo
    def external_validity(self,num_class, c):
        
        a1 = b1 = c1 = d1= 0
        for i in range(num_class):
            a1+= sum ( (self.count_class_cluster[i]-1)*self.count_class_cluster[i]/2 )
            b1+= (pow(sum(self.count_class_cluster[i]),2) - sum(pow(self.count_class_cluster[i],2)) )/2
        for i in range(num_class-1):
            for j in range(i+1, num_class):
                for k in range(c):
                    c1+=self.count_class_cluster[i][k] * self.count_class_cluster[j][k]
                    for h in range(c):
                        if h!=k :
                            d1+=self.count_class_cluster[i][k] * self.count_class_cluster[j][h]
        # Rand int
        w1 =  (a1 + d1)/ (a1 +b1 + c1 +d1)
        # Adjusted Rand Index
        M= a1 + b1 +c1 +d1
        w2 = (a1 - (a1+c1)*(a1+b1)/M) /( (2*a1+b1+c1)/2 - (a1+c1)*(a1+b1)/M )
        #Jaccard Coefficient
        w3 = a1 / (a1+b1+c1)
        self.w1=w1
        self.w2=w2
        self.w3=w3
        
    def set_dict_cluster(self):
        for i in range(self.num_class):
            temp = np.argmax(self.count_class_cluster[i])
            self.dict_cluster[self.label_list[i]] = temp
        return

    def rs_dict_cluster(self):
        self.dict_cluster = {}
        return
    
    def thuat_toan_1_pha(self,epsilon,m,c,k1):
        self.rs_dict_cluster()
        self.set_c(c)
        self.generate_U_ngang(k1)
        self.generate_V()

        Epsilon = np.zeros((c,self.p)) + epsilon
        i=1
        while True:
            # print("-----------------------\nChay lan "+ str(i)+"\n")
            self.U= self.cong_thuc_6(m)
            V_truoc =self.V
            self.V= self.cong_thuc_4(m)
            delta_V = abs(self.V- V_truoc)
            ktra = np.less_equal(delta_V, Epsilon)
            if (np.all(ktra)):
                break
        self.count_class(self.num_class, c)
        self.external_validity(self.num_class, c)
        return
        
    def thuat_toan_2_pha(self,epsilon,m,c,k1):
        # Pha 1 (Khong giam sat k1 = 0)
        self.thuat_toan_1_pha(epsilon,m,c,0)
        self.set_dict_cluster()  
        
        #Pha 2(Co giam sat k1 = k1)
        Epsilon = np.zeros((c,self.p)) + epsilon
        self.generate_U_ngang(k1)
        while True:
            self.U= self.cong_thuc_6(m)
        
            V_truoc = self.V
            self.V= self.cong_thuc_4(m)
        
            delta_V = abs(self.V- V_truoc)
            ktra = np.less_equal(delta_V, Epsilon)
            if (np.all(ktra)):
                break
        self.count_class(self.num_class, c)
        self.external_validity(self.num_class, c)
        self.rs_dict_cluster()
        return
    
    def freeMemory(self):
        self.U = None
        self.U_ngang = None
        self.c = None
        self.dict_cluster = None
        self.index_x_giamsat = None
        self.V = None