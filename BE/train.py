import pandas as pd
import sklearn, os, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import sqlite3, pickle
from fcm2 import FCM1

basedir = os.path.dirname((os.path.dirname(__file__)))
results_path = os.path.join(basedir, 'D:\Dongtacgia\dongtacgia\dongtacgia\Results')
models_path = os.path.join(basedir, 'D:\Dongtacgia\dongtacgia\dongtacgia\Models')
db_path = os.path.join(os.path.dirname(basedir), 'D:\Dongtacgia\dongtacgia\dongtacgia\Data_Project3')


def get_test_authors(data_name, test_percent):
    data_path = results_path + "/" + data_name
    data = pd.read_csv(data_path)
    data = data.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    X = data.drop(columns=['id_author_1', 'id_author_2', 'Label'])
    y = data['Label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_percent/100), random_state=608, shuffle=True)
    
    list_test_authors = set()
    list_CN = list(X_test['CommonNeighbor'])
    list_AA = list(X_test['AdamicAdar'])
    list_JC = list(X_test['JaccardCoefficient'])
    list_PA = list(X_test['PreferentialAttachment'])
    list_RA = list(X_test['ResourceAllocation'])
    list_SP = list(X_test['ShortestPath'])
    list_CC = list(X_test['CommonCountry'])

    for i in range(len(list_CN)):
        tmp_df = X_test[(X_test['iCommonNeighbor'] == list_CN[i]) &
                        (X_test['AdamicAdar'] == list_AA[i]) &
                        (X_test['JaccardCoefficient'] == list_JC[i]) &
                        (X_test['PreferentialAttachment'] == list_PA[i]) &
                        (X_test['ResourceAllocation'] == list_RA[i]) &
                        (X_test['ShortestPath'] == list_SP[i]) &
                        (X_test['CommonCountry'] == list_CC[i])
        ]
        for id1 in list(tmp_df['id_author_1']):
            list_test_authors.add(id1)
        for id2 in list(tmp_df['id_author_2']):
            list_test_authors.add(id2)
         
    list_id_names = []
    with sqlite3.connect(db_path + '/db.sqlite3') as conn:
        cur = conn.cursor()
        query = ("select id, first_name, last_name from collab_author \
                    where id in ({seq})"
                .format(seq=','.join(['?']*len(list_test_authors))))
        cur.execute(query, list_test_authors)
        result = cur.fetchall()
        for id, first_name, last_name in result:
            list_id_names.append((id, first_name + " " + last_name))
        return list_id_names

def train(data_name, test_percent):
    test_percent = int(test_percent)
    data_path = results_path + "/" + data_name
    print(data_path)
    data = pd.read_csv(data_path)
    data = data.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    au = data[['id_author_1', 'id_author_2']]
    X = data.drop(columns=['id_author_1', 'id_author_2', 'Label'])
    y = data['Label']
    print(X.shape)
    print("Tỉ lệ nhãn -1-1")
    print(np.sum(y==-1), end='--')
    print(np.sum(y==1))

    # scaler = MinMaxScaler().fit(X)
    # X = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_percent/100), random_state=608, shuffle=True)
    print('batdau')
    # print(X_train[:10])
    model1 = FCM1()
    # model = svm.SVC(kernel='rbf', max_iter=5000)
    # model.fit(X_train, y_train)
    # print(X_train[:5])
    model1.read_data(data=X,label=y,au=au)
    model1.preprocess_data()
    model1.thuat_toan_2_pha(1e-6,2,2,500)
    # y_pred = model.predict(X_test)
    print('da train xong')
    # print("Rand:",model1.w1)
    # print("AdjustRand:",model1.w2)
    # print("JC:",model1.w3)
    
    
    result = {}
    # result['rand'] = model1.w1
    # result['arand'] = model1.w2
    # result['jc'] = model1.w3
    

    tmp = data_name.split('_')
    
    tmp[0] = "Model"
    model_name = "_".join(tmp)[:-4] + ".pkl"
    with open(models_path + '/' + model_name, 'wb') as file:
        pickle.dump(model1, file)

    # tmp[0] = "Scaler"
    # scaler_name = "_".join(tmp)[:-4] + ".pkl"
    # with open(models_path + '/' + scaler_name, 'wb') as file:
    #     pickle.dump(scaler, file)


    return json.dumps({"results": result, "model_name": model_name})

