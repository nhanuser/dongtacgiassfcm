import sqlite3, json, os, pickle
from query import *
from co_author_graph import *
from define_scores import *
import numpy as np
from collections import defaultdict

basedir = os.path.dirname(os.path.dirname((os.path.dirname(__file__))))
db_path = os.path.join(basedir, 'D:/Dongtacgia/dongtacgia/dongtacgia/Data_Project3')
models_path = os.path.join(basedir, 'D:/Dongtacgia/dongtacgia/dongtacgia/Models')


def create_sub_db(topic, from_date, to_date):
    # _ = create_sub_tables([topic], from_date, to_date)
    sub_db_name = "subDB_" + topic + "_" + from_date + "_" + to_date
    return sub_db_name

def create_co_author_table(topic, from_date, to_date):
    _ = create_co_authors([topic], from_date, to_date)

def get_co_author(id, sub_db_name):   # find co authors of ONLY one author to save memory
    with sqlite3.connect(db_path + '/' + sub_db_name + ".sqlite3") as conn:
        cur = conn.cursor()
        query = ("select distinct pa.author_id from paper_authors pa \
                  where pa.paper_id in \
                  (select paper_id from paper_authors \
                   where author_id = " + str(id) + ") \
                  and pa.author_id != " + str(id)
                )
        cur.execute(query)
        co_authors = cur.fetchall()
        return co_authors
    
def get_potential(id, adj, sub_db_name):  # this could be extended to find candidates by affiliation
    potential = set()
    co_authors = list(adj[id].keys())
    for u in co_authors:
        u_co_authors = list(adj[u].keys())
        for v in u_co_authors:
            if v in co_authors or v == id:
                continue
            potential.add(v)
    result = []  # find candidates with same country_id
    with sqlite3.connect(db_path + '/' + sub_db_name + ".sqlite3") as conn:
        conn.row_factory = lambda cursor, row: row[0]
        cur = conn.cursor()
        query = ("select id from author \
                  where country_id = (select country_id from author where id = " + str(id) + ") \
                ")
        cur.execute(query)
        result = cur.fetchall()
        result = [i for i in result if i not in co_authors]
    for i in result:
        potential.add(i)
    return potential

def recommend(topic, from_date, to_date, author_id, model_name):
    list_recommend = defaultdict(int)
    # print("Creating sub network")
    sub_db_name = create_sub_db(topic, from_date, to_date)
    # print("Sub network created")
    create_co_author_table(topic, from_date, to_date)
    print("Creating co author network...")
    graph = Co_Author_Graph(db_path + '/' + sub_db_name + ".sqlite3")
    print("Co author network created")
    adj = graph.adj
    if author_id in adj:
        potential = get_potential(author_id, adj, sub_db_name)   # get candidates from network links
        scores = dict()
        # print("id:", author_id)
        # print("Potential:", potential)
        co_id = list(adj[author_id].keys())
        t = max(graph.time_patterns) + 1
        for u in potential:
            co_u = list(adj[u].keys())
            common_neighbors = list(set(co_id) & set(co_u))
            weight_type = "unweighted"
            cm = CommonNeighbor(author_id, u, adj, t, co_id, co_u, common_neighbors, weight_type)
            aa = AdamicAdar(author_id, u, adj, t, co_id, co_u, common_neighbors, weight_type)
            jc = JaccardCoefficient(author_id, u, adj, t, co_id, co_u, common_neighbors, weight_type)
            pa = PreferentialAttachment(author_id, u, adj, t, co_id, co_u, common_neighbors, weight_type)
            ra = ResourceAllocation(author_id, u, adj, t, co_id, co_u, common_neighbors, weight_type)
            sp = ShortestPath(author_id, u, adj, t)
            cc = CommonCountry_pair(author_id, u, graph.list_vertices, common_neighbors)
            score = np.array([cm,aa,jc,pa,ra,sp,cc])
            scores[u] = score

        X = np.array(list(scores.values()))
        # tmp = model_name.split("_")
        # tmp[0] = "Scaler"
        # scaler_name = "_".join(tmp)
        # print(scaler_name)
        # print(X[0:5])
        # print("-------------")
        # with open(models_path + '/' + scaler_name, 'rb') as file:
        #     scaler = pickle.load(file)
        #     X = scaler.transform(X)
        # print(X)
        with open(models_path + '/' + model_name, 'rb') as file:
            model = pickle.load(file)
            # y = model.predict(X)
            # print(y)
            # for idx,u in enumerate(potential):
            #     if y[idx] == 0:
            #         list_recommend[u] = np.mean(X[idx])
            au=model.au
            data=model.data
            c=model.cluster
            list_rec =potentials(author_id,au,data,c)
            list_recommend = {row[0]: row[1] for row in list_rec}
        list_recommend = dict(sorted(list_recommend.items(), key=lambda item: item[1], reverse=True))
        print(list_rec)
        print(list_recommend)
        # get name of candidates
        result = get_all_authors(topic, from_date, to_date)
        records = json.loads(result)
        mapping = {}
        for idx in range(len(records['id'])):
            mapping[int(records['id'][idx])] = records["first_name"][idx] + " " + records["last_name"][idx] 
        list_name = []
        for cand in list_recommend.keys():
            list_name.append(mapping[cand])
        return json.dumps({"potential": list(list_recommend.keys()), 'name': list_name, 'score': list(list_recommend.values())})
    else:
        return json.dumps({"potential": "Invalid author"})
            
def potentials(idx,au,data,c): #lay cac potential
    # Gộp các cột từ ba mảng
    dt = np.column_stack((au, data))
    result_array= np.column_stack((dt,c))

    df= result_array
    result_rows = df[(df[:, 0] == idx) | (df[:, 1] == idx)]
    array = np.empty((0, 2))

    #idx =1827147
    for u in result_rows: # lấy ra id author còn lại + mean các giá trị
        if (int(u[0]) ==idx):
            mean=float(np.mean(u[2:-2]))
            new_row = np.array([u[1], mean])
            array= np.vstack([array, new_row])
        elif  (u[1]==idx):
            mean=np.mean(u[2:-2])
            new_row = np.array([u[0], mean])
            array= np.vstack([array, new_row])
    return array

        
    

