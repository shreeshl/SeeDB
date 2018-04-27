from __future__ import division
from collections import defaultdict
import math
import psycopg2
from psycopg2.extensions import AsIs

def kl_divergence(p, q, normalizer):
    deviation = 0
    for k in p:
        if k not in q: continue
        if p[k]==0 or q[k] == 0 : continue
        pi, qi = p[k]/normalizer, q[k]/normalizer
        deviation += pi*math.log(pi/qi)
    return deviation

grp_attr = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
'sex','capital-gain','capital-loss','hours-per-week', 'native-country', 'salary']
functions = ['avg', 'sum', 'min', 'max', 'count']
measure_attr = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
'sex','capital-gain','capital-loss','hours-per-week', 'native-country', 'salary']

K = 5
delta = 1e-3
conn = psycopg2.connect("host=localhost dbname=shreesh user=shreesh")
cur = conn.cursor()
tables = {}
utility = defaultdict(float)
data = 'adult.data.txt'
file = open(data)
N = len(f)

for i, batch in enumerate(data):
    m = i + 1
    #Reinitialize D with batch
    if i>0:
        cur.execute("DROP TABLE d");
    
    cur.execute("""CREATE TABLE d (age INTEGER, workclass VARCHAR(200), fnlwgt INTEGER, education VARCHAR(200), education_num INTEGER, marital_status VARCHAR(30), occupation VARCHAR(50), relationship VARCHAR(50), race VARCHAR(50), sex VARCHAR(10), capital_gain INTEGER, capital_loss INTEGER, hours_per_week INTEGER, native_country VARCHAR(100), salary VARCHAR(50));""") #or update
    cur.copy_from(file, 'd', sep=",")
    
    for a in grp_attr:
        for m in measure_attr:
            for f in functions:
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
                    normalizer = 0.0
                    for i in view:
                        if i[2]==1: #MARRIED
                            p_dist[i[0]] = i[1]
                        else:       #UNMARRIED
                            q_dist[i[0]] = i[1]
                        normalizer += i[1]
                    
                    utility[a,f,m] += kl_divergence(p, q, normalizer)
                    epsilon_m = math.sqrt(  (1-(m-1)/N) * (2*math.log(math.log(m)) + math.log(math.pi.^2/(3*delta)))/(2*m))
                    lower_bound = epsilon_m - utility[a,f,m]/m
                    upper_bound = epsilon_m + utility[a,f,m]/m
                    tables[a,f,m] = (lower_bound, upper_bound)

    measure = tables.items()  #measure[0] = ((a,f,m), (lower_bound, upper_bound))
    measure.sort(key=lambda tup: -1*tup[1][1])
    if len(measure)>K:
        lowestLowerBound = measure[K-1][1]
        for view in measure[K:]:
            if view[2]<lowestLowerBound:
                del tables[view[0]]
    
    



