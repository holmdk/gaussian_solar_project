import pandas as pd

def nc_to_pd(data, name=None, save=True, save_folder=None):
    data = data.to_dataframe()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[['SIS']]

    data.dropna(inplace=True)

    if save == True:
        data.to_csv(save_folder + name + '.csv')

    return data


def prepare_simulated(data, time_idx):
    # we define timestamps as our index in utc time
    data['time'] = pd.to_datetime(data.time, utc=True)
    data.set_index('time', inplace=True)

    # we remove missing observations
    data = data[data['missing'] == 0]

    # And then we only select time and electricty as the columns
    data = data[['electricity']]

    # and finally, we select the timestamps defined for our satellite dataset
    data = data.loc[time_idx]

    return data