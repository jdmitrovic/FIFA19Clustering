import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


df = pd.read_csv('data.csv')
# print(df.head())

# df = df[df.Position != 'GK']

wages = df.loc[:, 'Wage']

new_wages = []

for w in wages:
    if pd.isna(w):
        new_wages.append(w)
    elif "M" in w:
        new_wages.append(float(w[1:len(w)-1]) * 1000000)
    elif "K" in w:
        new_wages.append(float(w[1:len(w) - 1]) * 1000)
    elif w != "":
        new_wages.append(float(w[1:]))
    else:
        new_wages.append(0.0)

df['Wage'] = new_wages
# print(df.head().loc[:, "Wage_new"])


value = df.loc[:, 'Value']

new_values = []

for v in value:
    if pd.isna(v):
        new_values.append(v)
    elif "M" in v:
        new_values.append(float(v[1:len(v)-1]) * 1000000)
    elif "K" in v:
        new_values.append(float(v[1:len(v) - 1]) * 1000)
    elif v != "":
        new_values.append(float(v[1:]))
    else:
        new_values.append(0.0)

df['Value'] = new_values
# print(df.head().loc[:, 'Value'])

clauses = df.loc[:, 'Release Clause']

new_clauses = []

for c in clauses:
    if pd.isna(c):
        new_clauses.append(c)
    elif "M" in c:
        new_clauses.append(float(c[1:len(c)-1]) * 1000000)
    elif "K" in c:
        new_clauses.append(float(c[1:len(c) - 1]) * 1000)
    else:
        new_clauses.append(float(c[1:]))

df['Release Clause'] = new_clauses

for position in ['CB', 'RB', 'RWB', 'CDM', 'CM', 'CAM', 'RM', 'RW', 'RF', 'ST', 'CF']:
    pos_rating = df.loc[:, position]

    new_rating = []

    for p in pos_rating:
        if pd.isna(p):
            new_rating.append(p)
        elif "+" in p:
            new_rating.append(float(p[0:2]) + float(p[3:]))
        else:
            new_rating.append(float(p))

    df[position] = new_rating

del df['LB']
del df['LWB']
del df['LM']
del df['RAM']
del df['LAM']
del df['LW']
del df['LF']
del df['LS']
del df['RS']
del df['RDM']
del df['LDM']
del df['LCB']
del df['RCB']
del df['RCM']
del df['LCM']

new_weights = []
weights = df.loc[:, 'Weight']

for w in weights:
    if pd.isna(w):
        new_weights.append(w)
    else:
        new_weights.append(float(w[0:len(w)-3]) * 0.45359237)

df['Weight'] = new_weights

new_heights = []
heights = df.loc[:, 'Height']

for h in heights:
    if pd.isna(h):
        new_heights.append(h)
    else:
        new_heights.append(float(h[0]) * 12 * 2.54 + float(h[2:]) * 2.54)

df['Height'] = new_heights

del df['Photo']
del df['Flag']
del df['Club Logo']
del df['Joined']
del df['ID']
del df['Unnamed: 0']
del df['Loaned From']

del df['Name']
del df['Contract Valid Until']

# pd.set_option('display.max_rows', None)
# print(df.dtypes)

for a in ['Club', 'Nationality', 'Preferred Foot', 'Work Rate',
          'Body Type', 'Real Face', 'Position']:
    df[a] = df[a].astype('category')
    df[a] = df[a].cat.codes

features = df.columns

scaler = MinMaxScaler().fit(df[features])
scaled_df = pd.DataFrame(scaler.transform(df[features]))
scaled_df.columns = features

scaled_df.fillna(scaled_df.mean(), inplace=True)

fig = plt.figure(figsize=(6.5, 6.5))

index = 1

colors = ['red', 'blue', 'green', 'gold', 'm', 'black', 'brown']

for link in ['ward', 'complete', 'average', 'simple']:
    for num_clusters in range(2, 6):
        est = AgglomerativeClustering(n_clusters=num_clusters, linkage=link, affinity='euclidean')
        est.fit(scaled_df)
        df['labels'] = est.labels_

        sp = fig.add_subplot(2, 2, index)
        sp.set_xlabel('Ball Control')
        sp.set_ylabel('Interceptions')

        for j in range(0, num_clusters):
            cluster = df.loc[df['labels'] == j]
            plt.scatter(cluster['BallControl'], cluster['Interceptions'], color=colors[j], label="cluster %d" % j)

        # plt.title('linkage %s' % link)
        # plt.legend()

        print("Silhouette score: %f " % silhouette_score(scaled_df, est.labels_))

        index += 1

#
# plt.tight_layout()
# plt.show()

fig.savefig('demo.png', bbox_inches='tight')
