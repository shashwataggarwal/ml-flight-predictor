import datetime
import pandas as pd
import numpy as np
import re
import os.path
from tqdm import tqdm


def get_time_delta(time):
    """
    Returns the time delta in minutes

    Input: Time of format `%H hrs %M mins ` or `%H hrs `
    """
    hrs_mins = re.findall(r'\d+', time)

    assert len(hrs_mins) > 0 and len(hrs_mins) < 3

    hours = int(hrs_mins[0])
    minutes = 0
    if len(hrs_mins) == 2:
        minutes = int(hrs_mins[1])

    time_delta = datetime.timedelta(hours=hours, minutes=minutes).seconds
    return time_delta/60


def get_part_of_day(hour):
    """
    Returns part of day for given hour

    Input:  hour (int)
    Output: Part of day (str) [morning (5 to 11), afternoon (12 to 17), evening (18 to 22), night (23 to 4)]
    """
    return (
        "morning" if 5 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 17
        else
        "evening" if 18 <= hour <= 22
        else
        "night"
    )


def classify_time(time):
    """
    Classify given time to part of day

    Input:  time in `HH:MM` format (str)
    Output: Part of day (str) [morning (5 to 11), afternoon (12 to 17), evening (18 to 22), night (23 to 4)]
    """
    hour = datetime.datetime.strptime(time, "%H:%M").time().hour
    return get_part_of_day(hour)


def clean_number_of_stops(stop):
    if (stop[0] == 'N'):
        return 0
    else:
        return int(stop[0])


def clean_price(price):

    return int(re.sub(r'[^\d.]', '', price))


def days_between(df):
    d1 = datetime.datetime.strptime(df['booking_date'], "%Y-%m-%d")
    d2 = datetime.datetime.strptime(df['departure_date'], "%Y-%m-%d")
    return abs((d2 - d1).days)


def combine_input_files(path_to_dir='./Data'):
    '''
    Return combined data set for all files

    Input: path to directory
    '''
    startDate = datetime.datetime(2020, 11, 8)
    df = pd.DataFrame()
    files = [f for f in os.listdir(path_to_dir)
             if os.path.isfile(os.path.join(path_to_dir, f))]
    for i, file_name in enumerate(files):
        temp = pd.read_csv('{}/{}'.format(path_to_dir, file_name), header=0)
        if (i == 0):
            df = temp
        else:
            df = pd.concat([df, temp], axis=0)

    return df


def hot_encode_flight_path(df):
    '''
    Return updated dataframe with one hot encoded flight paths

    Input: dataframe with 'departure_city' & 'arrival_city' as columns
    '''
    df['flight_path'] = df['departure_city'].str.cat(
        df['arrival_city'], sep="-")
    hot_encoding = pd.get_dummies(df['flight_path'])
    df.drop(columns=['departure_city', 'arrival_city'], inplace=True)
    df = pd.concat([df, hot_encoding], axis=1)
    return df


def hot_encode_clusters(df):
    '''
    Return updated dataframe with one hot encoded clusters based on departure_time[enum(morning, afternoon, evening, night)] and day of departure

    Input: dataframe with 'departure_time' & 'departure_day' as columns
    '''
    df['departure_day'] = df['departure_day'].astype(str)
    df['departure_time_day'] = df['departure_time'].str.cat(
        df['departure_day'], sep="-")
    hot_encoding = pd.get_dummies(df['departure_time_day'])
    df = pd.concat([df, hot_encoding], axis=1)
    df['departure_day'] = df['departure_day'].astype(int)
    return df


def hot_encode_airline(df):
    '''
    Return updated dataframe with one hot encoded airline

    Input: dataframe with 'airline' as column
    '''
    hot_encoding = pd.get_dummies(df['airline'])
    df = pd.concat([hot_encoding, df], axis=1)
    return df


def hot_encode_days(df):
    '''
    Return updated dataframe with one hot encoded days

    Input: dataframe with 'booking_day', 'departure_day' as column
    '''
    hot_encoding_bd = pd.get_dummies(df['booking_day'], prefix='bd_')
    hot_encoding_dd = pd.get_dummies(df['departure_day'], prefix='dd_')
    df = pd.concat([df, hot_encoding_bd, hot_encoding_dd], axis=1)
    return df


def drop_columns(df):
    '''
    Return updated dataframe with dropped columns

    Input: dataframe with 'flight_code', 'flight_path', 'departure_day', 'booking_day', 'airline', 'departure_date', 'booking_date', 'departure_time', 'arrival_time', 'departure_time_day' as columns
    '''
    df.drop(columns=['flight_code', 'flight_path',
                     'departure_day', 'booking_day', 'airline', 'departure_date', 'booking_date', 'departure_time', 'arrival_time', 'departure_time_day'], inplace=True)
    return df


def drop_pairs_airlines(df):
    air_list = ['Air India', 'IndiGo', 'Spicejet',
                'Vistara', 'Go Air', 'AirAsia']
    df = df.loc[df['airline'].isin(air_list)]
    return df


def removeOutliers(df):
    df = df[df['flight_cost'] < 11000]
    return df

def createPivotTable(df):
    # Create groupby ref
    groups = df.groupby(['airline','flight_path','days_to_depart'])
    
    # Weights and parameters
    prev_range_date = 5
    max_day_to_depart = df['days_to_depart'].max()
    weight = 0.75

    pivot_frame = pd.DataFrame([],columns=['airline','flight_path','days_to_depart','count','mean','min', 'std','first_quartile','custom_fare'])

    # Iterate over groups
    for indexes, group in tqdm(groups):
        # Calculaate aggregates
        count = group['flight_cost'].count()
        avg = group['flight_cost'].mean()
        min_fare = group['flight_cost'].min()
        std_dev = group['flight_cost'].std()
        group_quartile = np.percentile(group['flight_cost'], 25).astype('float')
        # entire_quartile = np.percentile(df[(df['airline']==indexes[0]) & (df['flight_path']==indexes[1]) & (df['days_to_depart'] >= indexes[2])]['flight_cost'], 25)

        # Handling Edge cases
        if (indexes[2]>=max_day_to_depart-prev_range_date):
            temp_frame = pd.DataFrame([*indexes,count,avg,min_fare,std_dev,group_quartile,group_quartile]).T

        else:
            prev_group_quartile = np.percentile(df[(df['airline']==indexes[0]) & (df['flight_path']==indexes[1]) & (df['days_to_depart'] == indexes[2]+prev_range_date)]['flight_cost'], 25)
            # prev_range_quartile = np.percentile(df[(df['airline']==indexes[0]) & (df['flight_path']==indexes[1]) & (df['days_to_depart'] >= indexes[2]+prev_range_date)]['flight_cost'], 25)
            custom_fare = (weight * group_quartile) + ((1-weight) * prev_group_quartile)
            temp_frame = pd.DataFrame([*indexes, count,avg,min_fare,std_dev,group_quartile,custom_fare]).T
        
        # append to pivot table
        temp_frame.columns = ['airline','flight_path','days_to_depart','count','mean','min', 'std','first_quartile','custom_fare']
        pivot_frame = pivot_frame.append(temp_frame, ignore_index=True)

    # Creating Label variable
    pivot_frame['logical'] = np.zeros(pivot_frame.shape[0])

    # Creating group for assigning labels
    temp_group = pivot_frame.groupby(['airline','flight_path'])

    # Iterating and labeling
    for indexes, res in tqdm(temp_group):
        pivot_frame.loc[(pivot_frame['airline']==indexes[0]) & (pivot_frame['flight_path']==indexes[1]) & (pivot_frame['custom_fare']==res['custom_fare'].min()),'logical'] = 1

    pivot_frame.to_csv('./Processed_Data/pivot_data.csv',index=False)
    return pivot_frame

def createClasification(df):
    # Get pivot table
    pivot_frame = createPivotTable(df)

    # Create new feature column
    df['label'] = np.zeros(df.shape[0])
    df['group_count'] = np.zeros(df.shape[0])
    df['group_mean'] = np.zeros(df.shape[0])
    df['group_min'] = np.zeros(df.shape[0])
    df['group_quartile'] = np.zeros(df.shape[0])
    df['custom_fare'] = np.zeros(df.shape[0])

    # Fill feature columns data
    for i in tqdm(range(pivot_frame.shape[0])):
        entry = pivot_frame.iloc[i]
        df.loc[(df['airline']==entry['airline']) & (df['flight_path']==entry['flight_path']) & (df['days_to_depart'] == entry['days_to_depart']),'label']=entry['logical_1']
        df.loc[(df['airline']==entry['airline']) & (df['flight_path']==entry['flight_path']) & (df['days_to_depart'] == entry['days_to_depart']),'group_count']=entry['count']
        df.loc[(df['airline']==entry['airline']) & (df['flight_path']==entry['flight_path']) & (df['days_to_depart'] == entry['days_to_depart']),'group_mean']=entry['mean']
        df.loc[(df['airline']==entry['airline']) & (df['flight_path']==entry['flight_path']) & (df['days_to_depart'] == entry['days_to_depart']),'group_min']=entry['min']
        df.loc[(df['airline']==entry['airline']) & (df['flight_path']==entry['flight_path']) & (df['days_to_depart'] == entry['days_to_depart']),'group_quartile']=entry['first_quartile']
        df.loc[(df['airline']==entry['airline']) & (df['flight_path']==entry['flight_path']) & (df['days_to_depart'] == entry['days_to_depart']),'custom_fare']=entry['custom_fare_1']

    return df

def preprocess(graphs=False):
    '''
    Return preprocessed Dataset

    Input: boolean flag for graphs
    '''

    # Import Dataset
    df = combine_input_files()

    df = df[df['airline'] != 'airline']

    df['flight_duration'] = df['flight_duration'].apply(get_time_delta)
    df['departure_time'] = df['departure_time'].apply(classify_time)

    df['number_of_stops'] = df['number_of_stops'].apply(clean_number_of_stops)
    df['flight_cost'] = df['flight_cost'].apply(clean_price)
    df['days_to_depart'] = df.apply(days_between, axis=1)

    # Can plot graphs now
    # if(graphs):
    df = drop_pairs_airlines(df)

    # One Hot encoding
    df = hot_encode_flight_path(df)
    df = hot_encode_airline(df)
    df = hot_encode_days(df)
    df = hot_encode_clusters(df)

    df.drop_duplicates(inplace=True)
    df = createClasification(df)
    
    if (not graphs):
        # Drop columns
        df = drop_columns(df)
        df = removeOutliers(df)
        df.to_csv('./Processed_Data/training_data.csv',index=False)
        
    else:
        df.to_csv('./Processed_Data/graph_data.csv',index=False)

    return df

def loadData(path='./Processed_Data', file_name='training_data.csv'):
    df = pd.read_csv('{}/{}'.format(path,file_name), header=0)
    return df

if __name__ == "__main__":
    df = preprocess()
    print(df.head())