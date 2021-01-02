import numpy as np
import pandas as pd

#-------------------------------------------------------------------------------
# Functions

def save_aps_to_excel(excel_path, metrics_thresholds, aps):

    all_df = []

    # Creating dataframes
    for aps_name in aps.keys():

        # Accessing the aps (y axis)
        y = np.mean(aps[aps_name], axis=0).reshape((-1,1))

        # Accesing the aps (x axis)
        x = metrics_thresholds[aps_name].reshape((-1,1))

        # Creating data to be place in the excel
        data = np.hstack((x,y))

        # Creating dataframe
        df = pd.DataFrame(data, columns=[f'{aps_name} - x', f'{aps_name} - y'])

        # Storing data
        all_df.append(df)

    # Store dataframe into excel file
    with pd.ExcelWriter(excel_path) as writer:
        aps_names = list(aps.keys())
        for i, df in enumerate(all_df):
            df.to_excel(writer, sheet_name=aps_names[i])