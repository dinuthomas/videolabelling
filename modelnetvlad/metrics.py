import pandas as pd

df = pd.read_csv('./results.csv')
df.fillna('', inplace=True)
df = df[df['vertical1']!='']

def drop_duplicates(row):
    times = {}
    for i in row['vertical1'].split(','):
        for v in row['value'].split(' '):
            if i not in times.keys():
                times[i] = float(v) if v !='' else 0
            else:
                times[i] += float(v) if v !='' else 0
    return sorted(times.items(), key = lambda x:x[1], reverse=True)

def drop_duplicates_2(row):
    vertical = []
    for i in row['vertical1'].split(','):
        if i not in vertical:
            vertical.append(i)
    return vertical

def drop_duplicates_3(row):
    times = {}
    for i in row['vertical1'].split(','):
        if i not in times.keys():
            times[i] = 1 
        else:
            times[i] += 1
    return sorted(times.items(), key = lambda x:x[1], reverse=True)

def hit(row, k):
    hit = 0
    res = [w[0] for w in row['vertical']][:k]
    #res = row['vertical'][:k]
    for j in res:
        if j == row['category']: 
            hit = 1
            break
    return hit
        
#df['vertical'] = df.apply(drop_duplicates, axis=1)
df['vertical'] = df.apply(drop_duplicates_3, axis=1)
df['hit@1'] = df.apply(lambda row: hit(row, 1), axis=1)
df['hit@3'] = df.apply(lambda row: hit(row, 3), axis=1)
df['hit@5'] = df.apply(lambda row: hit(row, 5), axis=1)

print(df[['hit@1', 'hit@3', 'hit@5']].mean())
hit_df = df[['category', 'hit@1', 'hit@3', 'hit@5']].groupby('category').sum().reset_index()
print(hit_df)
hit_df.to_csv('hit_ratio_3.csv')
#df.to_csv('hit_ratio.csv')