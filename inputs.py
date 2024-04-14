import pandas as pd
from models import Client, Depot

customers_df = pd.read_excel('./data_PTV_Fil_rouge/2_detail_table_customers.xls')
vehicles_df = pd.read_excel('./data_PTV_Fil_rouge/3_detail_table_vehicles.xls')
depots_df = pd.read_excel('./data_PTV_Fil_rouge/4_detail_table_depots.xls')[
        ['DEPOT_CODE', 'DEPOT_LONGITUDE', 'DEPOT_LATITUDE']
    ].drop_duplicates()

vehicle_weight = vehicles_df.VEHICLE_TOTAL_WEIGHT_KG.mean()


customers = [row for row in
             customers_df.groupby(
                 [
                     'CUSTOMER_CODE',
                     'CUSTOMER_LONGITUDE',
                     'CUSTOMER_LATITUDE'
                 ],
                 as_index=False
             ).aggregate(
                {'TOTAL_WEIGHT_KG': 'mean'}
             ).drop('CUSTOMER_CODE', axis=1).itertuples(index=False)
             ]

depots = [Depot(row['DEPOT_CODE'], row['DEPOT_LONGITUDE'], row['DEPOT_LATITUDE']) for row in depots_df.to_dict('records')]
