import preprocess

df = preprocess.loadData(file_name='graph_data_new.csv')
df = preprocess.drop_columns(df)
df = preprocess.removeOutliers(df)
df.to_csv('./Processed_Data/training_data_new.csv',index=False)