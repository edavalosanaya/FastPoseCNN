import numpy as np
import torch
import pandas as pd

#-------------------------------------------------------------------------------
# Functions

def save_aps_to_excel(excel_path, metrics_thresholds, aps):

    all_df = []

    # Creating dataframes
    for aps_name in aps.keys():

        # Classes and mean title
        y_columns = list(aps[aps_name].keys())

        # Accessing the aps (y axis)
        y = torch.stack(list(aps[aps_name].values())).t()

        # Accesing the aps (x axis)
        x = metrics_thresholds[aps_name].reshape((-1,1))

        # Creating data to be place in the excel
        data = torch.hstack((x,y))

        # Creating dataframe
        df = pd.DataFrame(data.cpu().numpy(), columns=[f'{aps_name} - x'] + y_columns)

        # Setting the index to be the x value
        df = df.set_index(f'{aps_name} - x')

        # Storing data
        all_df.append(df)

    # Store dataframe into excel file
    with pd.ExcelWriter(excel_path) as writer:
        aps_names = list(aps.keys())
        for i, df in enumerate(all_df):
            df.to_excel(writer, sheet_name=aps_names[i])