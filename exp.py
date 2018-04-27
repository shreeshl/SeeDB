from collections import defaultdict
import math
import psycopg2
from psycopg2.extensions import AsIs

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
    data_each_file = len(f)/no_files
    new_filename = 'adult.data'
    for i in range(len(f)): 
        if i%data_each_file==0:
            try:
                new_file.close()
            except:
                pass    
            if n==len(f) : return
            new_file = open(new_filename+'_'+str(i/data_each_file)+'.txt', 'w')    
        new_file.write(f[i])
    return

#education_num, marital_status is out
grp_attr = ['age','workclass','fnlwgt','education','occupation','relationship','race',
'sex','capital_gain','capital_loss','hours_per_week', 'native_country', 'salary']
functions = ['avg', 'sum', 'min', 'max', 'count']
measure_attr = ['age','capital_gain','capital_loss','hours_per_week']

K = 5
delta = 1e-3
conn = psycopg2.connect("host=localhost dbname=shreesh user=shreesh")
cur = conn.cursor()
tables = {}
utility = defaultdict(float)
N = len(f)
no_files = 10
create_n_files('adult.data.txt', no_files)

for i in range(no_files):
    M = i + 1
    #Reinitialize D with batch
    if i>0:
        cur.execute("DROP TABLE d;")
    
    cur.execute("""CREATE TABLE d (age INTEGER, workclass VARCHAR(200), fnlwgt INTEGER, education VARCHAR(200), education_num INTEGER, marital_status VARCHAR(30), occupation VARCHAR(50), relationship VARCHAR(50), race VARCHAR(50), sex VARCHAR(10), capital_gain INTEGER, capital_loss INTEGER, hours_per_week INTEGER, native_country VARCHAR(100), salary VARCHAR(50));""") #or update
    data = 'adult.data_' + str(i) + '.txt'
    file = open(data)
    cur.copy_from(file, 'd', sep=",")
    
    for a in grp_attr:
        for m in measure_attr:
            for f in functions:
                if a==m : continue
                if (a,f,m) in tables or i==0:
            
                    cur.execute("""SELECT %s, %s(%s),
                        CASE marital_status
                        when 'Married-civ-spouse' then 1
                        when 'Married-spouse-absent' then 1
                        when 'Married-civ-spouse' then 1
                        ELSE 0
                        END as g1, 1 AS g2 FROM d GROUP BY %s, g1, g2;""", (AsIs(a),AsIs(f),AsIs(m), AsIs(a)))
                    #Measure utility and update tables
                    view = cur.fetchall()
                    p_dist = defaultdict(float)
                    q_dist = defaultdict(float)
                    normalizer_p, normalizer_q  = 0.0, 0.0
                    for x in view:
                        if x[2]==1: #MARRIED
                            p_dist[x[0]] = float(x[1])
                            normalizer_p += float(x[1])
                        else:       #UNMARRIED
                            q_dist[x[0]] = float(x[1])
                            normalizer_q += float(x[1])
                    
                    utility[a,f,m] += kl_divergence(p_dist, q_dist, normalizer_p, normalizer_q)
                    epsilon_m = math.sqrt(  (1-(M-1)/N) * (2*math.log(math.log(M)) + math.log(math.pi**2/(3*delta)))/(2*M))
                    lower_bound = epsilon_m - utility[a,f,m]/M
                    upper_bound = epsilon_m + utility[a,f,m]/M
                    tables[a,f,m] = (lower_bound, upper_bound)

    
    measure = tables.items()  #measure[0] = ((a,f,m), (lower_bound, upper_bound))
    measure.sort(key=lambda tup: -1*tup[1][1])
    if len(measure)>K:
        lowestLowerBound = measure[K-1][1][0]
        for v in measure[K:]:
            if v[1][0]<lowestLowerBound:
                del tables[v[0]]
    
    



