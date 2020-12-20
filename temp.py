import preprocess

df = preprocess.loadData(file_name='graph_classification.csv')
df = preprocess.drop_columns(df)
df = preprocess.removeOutliers(df)
df.to_csv('./Processed_Data/training_classification.csv',index=False)