from __future__ import division
from collections import defaultdict, OrderedDict
import math
import psycopg2
import time
from psycopg2.extensions import AsIs
import numpy as np
import matplotlib.pyplot as plt

class MyDict(defaultdict):
    def __missing__(self, key):
        value = [0, 0]
        self[key] = value
        return value

def kl_divergence(p, q, normalizer_p, normalizer_q):
    deviation = 0
    for k in p:
        if k not in q: continue
        if p[k]==0 or q[k] == 0 : continue
        pi, qi = p[k]/normalizer_p, q[k]/normalizer_q
        deviation += pi*math.log(pi/qi)
    return deviation

def create_n_files(filename, no_files):
    f = open(filename).read().split('\n')
    data_each_file = math.ceil(len(f)/no_files)
    new_filename = 'adult.data'
    for i in range(len(f)): 
        if i%data_each_file==0:
            try:
                new_file.write(f[i])
                new_file.close()
            except:
                pass    
            if i==len(f) : return
            new_file = open(new_filename+'_'+str(int(i/data_each_file))+'.txt', 'w')    
            continue
        
        new_file.write(f[i])
        if i!=len(f)-1 : new_file.write('\n')
    return len(f)

#education_num, marital_status is out
grp_attr = ['age','workclass','fnlwgt','education','occupation','relationship','race',
'sex','capital_gain','capital_loss','hours_per_week', 'native_country', 'salary']
functions = ['avg', 'sum', 'min', 'max', 'count']
measure_attr = ['age','capital_gain','capital_loss','hours_per_week']

K = 5
delta = 1e-3
no_files = 10
N = create_n_files('adult.data.txt', no_files)

conn = psycopg2.connect("host=localhost dbname=shreesh user=shreesh")
cur = conn.cursor()
tables = {}
utility = defaultdict(float)

tic = time.time()
for i in range(no_files):
    M = i + 1
    #Reinitialize D with batch
    if i>0:
        cur.execute("DROP TABLE d;")
    
    cur.execute("""CREATE TABLE d (age INTEGER, workclass VARCHAR(200), fnlwgt INTEGER, education VARCHAR(200), education_num INTEGER, marital_status VARCHAR(30), occupation VARCHAR(50), relationship VARCHAR(50), race VARCHAR(50), sex VARCHAR(10), capital_gain INTEGER, capital_loss INTEGER, hours_per_week INTEGER, native_country VARCHAR(100), salary VARCHAR(50));""") #or update
    data = 'adult.data_' + str(i) + '.txt'
    file = open(data)
    cur.copy_from(file, 'd', sep=",")
    # print i
    for a in grp_attr:
        for m in measure_attr:
            for f in functions:
                # print i, a, f, m
                if a==m : continue
                if (a,f,m) in utility or i==0:
                    # print i, a, f, m
                    cur.execute("""SELECT %s, %s(%s),
                        CASE marital_status
                        when 'Married-civ-spouse' then 1
                        when 'Married-spouse-absent' then 1
                        when 'Married-civ-spouse' then 1
                        ELSE 0
                        END as g1, 1 AS g2 FROM d GROUP BY %s, g1, g2;""", (AsIs(a),AsIs(f),AsIs(m), AsIs(a)))
                    #Measure utility and update tables
                    view = cur.fetchall()
                    p_dist = OrderedDict()
                    q_dist = OrderedDict()
                    normalizer_p, normalizer_q  = 0.0, 0.0
                    for x in view:
                        if x[2]==1: #MARRIED
                            p_dist[x[0]] = float(x[1])
                            normalizer_p += float(x[1])
                        else:       #UNMARRIED
                            q_dist[x[0]] = float(x[1])
                            normalizer_q += float(x[1])
                    
                    utility[a,f,m] += kl_divergence(p_dist, q_dist, normalizer_p, normalizer_q)
                    if M<=2: continue
                    epsilon_m = math.sqrt(  (1-(M-1)/N) * (2*math.log(math.log(M)) + math.log(math.pi**2/(3*delta)))/(2*M) )
                    lower_bound = utility[a,f,m]/M - epsilon_m 
                    upper_bound = utility[a,f,m]/M + epsilon_m
                    tables[a,f,m] = (lower_bound, upper_bound)

    
    measure = tables.items()  #measure[0] = ((a,f,m), (lower_bound, upper_bound))
    measure.sort(key=lambda tup: -1*tup[1][1])
    if len(measure)>K:
        lowestLowerBound = float('inf')
        for i in measure[:K]:
            lowestLowerBound = min(lowestLowerBound,i[1][0])
        for v in measure[K:]:
            if v[1][1]<lowestLowerBound:
                del tables[v[0]]    
                del utility[v[0]]
    print 'Time Taken for phase %d : %.3f sec'%(M, time.time()-tic)
    print 'Tables currently being considered : %d'%(len(utility))

print 'Total Time Taken : %.3f sec'%(time.time()-tic)


cur.execute("DROP TABLE d;")
cur.execute("""CREATE TABLE d (age INTEGER, workclass VARCHAR(200), fnlwgt INTEGER, education VARCHAR(200), education_num INTEGER, marital_status VARCHAR(30), occupation VARCHAR(50), relationship VARCHAR(50), race VARCHAR(50), sex VARCHAR(10), capital_gain INTEGER, capital_loss INTEGER, hours_per_week INTEGER, native_country VARCHAR(100), salary VARCHAR(50));""")
data = 'adult.data.txt'
file = open(data)
cur.copy_from(file, 'd', sep=",")
for key in utility:
    a,f,m = key
    cur.execute("""SELECT %s, %s(%s),
                    CASE marital_status
                    when 'Married-civ-spouse' then 1
                    when 'Married-spouse-absent' then 1
                    when 'Married-civ-spouse' then 1
                    ELSE 0
                    END as g1, 1 AS g2 FROM d GROUP BY %s, g1, g2;""", (AsIs(a),AsIs(f),AsIs(m), AsIs(a)))
    
    view = cur.fetchall()
    data_stats = MyDict()
    
    for x in view:
        if x[2]==1: #MARRIED
            data_stats[x[0]][0] = float(x[1])
        else:       #UNMARRIED
            data_stats[x[0]][1] = float(x[1])
    # all_keys = set(married_stats.keys() + unmarried_stats.keys())
    # for i in all_keys:
    #     if i not in married_stats:
    #         married_stats[i] = 0
    #     if i not in unmarried_stats:
    #         unmarried_stats[i] = 0
    
    objects = data_stats.keys()
    ind = np.arange(len(objects))
    values = np.array(data_stats.values())
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, values[:,0], width)
    rects2 = ax.bar(ind + width, values[:,1], width)
    
    ax.set_ylabel('%s'%f)
    ax.set_title('%s of %s grouped by %s'%(f,m,a))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(married_stats.keys(), rotation=20)

    ax.legend((rects1[0], rects2[0]), ('Married', 'Unmarried'))
    plt.savefig("%s_%s_%s.png"%(a,f,m))
    