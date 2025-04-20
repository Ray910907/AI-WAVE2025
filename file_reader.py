import pandas as pd
import numpy as np


class file_reader:
    def __init__(self, path):
        self.pth = {
            'acc':path + '(Train)ACCTS_Data_202412.csv',
            'eccus':path + '(Train)ECCUS_Data_202412.csv',
            'id':path + '(Train)ID_Data_202412.csv',
            'trsac':path + '(Train)SAV_TXN_Data_202412.csv',
        }
        self.today = 18290

    def read_merge_info(self):
        '''
            Read and merge account and transaction information.
        '''

        # Customizable
        merge_processed = {
            'TX_DATE': lambda x: (x.max()-x.min())/len(x) if len(x) > 1 else 30,
        }
        # Customizable
        fill_na = {
            'TX_DATE': 30,
            'TRN_COUNT': 0,
        }


        acc_info, label, id2index = self.read_account_info()
        transaction_info = self.read_transaction_info(id2index)

        # Process transaction information
        transaction_info = transaction_info.sort_values(by=['CUST_ID', 'TX_DATE'])
        transaction_aggregated = transaction_info.groupby('CUST_ID').agg(merge_processed).reset_index(drop=True)
        transaction_aggregated['CUST_ID'] = transaction_info['CUST_ID'].unique()
        remaining_columns = set(transaction_info.columns) - {'CUST_ID'} - set(merge_processed.keys())
        for col in remaining_columns:
            transaction_aggregated[col] = transaction_info.groupby('CUST_ID')[col].mean().reset_index(drop=True)

        # Number of transactions
        transaction_aggregated['TRN_COUNT'] = transaction_info.groupby('CUST_ID').size().reset_index(drop=True)

        # Merge
        acc_info = pd.merge(acc_info, transaction_aggregated, on='CUST_ID', how='left')


        #fill NAN
        acc_info = acc_info.fillna(fill_na)
        for col in remaining_columns:
            acc_info[col] = acc_info[col].fillna(0)

        return acc_info, label, id2index

    def read_account_info(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        '''
            Read account information and merge with ID information.
            Label is 1 if the account is in eccus, 0 otherwise.
        '''
        acc = pd.read_csv(self.pth['acc']).drop(columns=['ACCT_NBR'], errors='ignore')
        acc['ACCT_OPEN_DT'] = self.today - acc['ACCT_OPEN_DT']
        id = pd.read_csv(self.pth['id'])
        merged_data = pd.merge(acc, id, on='CUST_ID', how='left')

        all_ids = merged_data['CUST_ID'].unique()
        id2index = {cid: idx for idx, cid in enumerate(all_ids)}

        merged_data = merged_data.fillna(0)
        merged_data['CUST_ID'] = merged_data['CUST_ID'].map(id2index)
        merged_data = merged_data.dropna(subset=['CUST_ID'])
        merged_data['CUST_ID'] = merged_data['CUST_ID'].astype(int)
        merged_data = merged_data.sort_values(by='CUST_ID') 

        eccus = pd.read_csv(self.pth['eccus']).drop(columns=['ACCT_NBR'], errors='ignore')
        eccus['CUST_ID'] = eccus['CUST_ID'].map(id2index)
        label = merged_data['CUST_ID'].isin(eccus['CUST_ID']).astype(int)
        
        return merged_data, label, id2index
        

    def read_transaction_info(self,id2index) -> pd.DataFrame:
        '''
            Read transaction information and process it.
        '''
        # Customizable
        colum_proccessed = {
            'ACCT_NBR':'drop',
            'CUST_ID':'none',
            'TX_DATE':'turn_to_days',
            'TX_TIME':'classify',
            'DRCR':'one_hot_encode',
            'TX_AMT':'none',
            'PB_BAL':'none',
            'OWN_TRANS_ACCT':'drop',
            'OWN_TRANS_ID':'classify',
            'CHANNEL_CODE':'one_hot_encode',
            'TRN_CODE':'one_hot_encode',
            'BRANCH_NO':'drop',
            'EMP_NO':'drop',
            'mb_check':'none',
            'eb_check':'none',
            'SAME_NUMBER_IP':'none',
            'SAME_NUMBER_UUID':'none',
            'DAY_OF_WEEK':'classify'
        }

        classify_dict = {
            'TX_TIME': [[0,1,2,3,4,5,6,20,21,22,23]],
            'OWN_TRANS_ID': [['ID99999']],
            'DAY_OF_WEEK': [['Saturday', 'Sunday']]
        }

        trsac = pd.read_csv(self.pth['trsac'])
        for col in colum_proccessed.keys():
            if colum_proccessed[col] == 'drop':
                trsac = trsac.drop(columns=[col])
            elif colum_proccessed[col] == 'turn_to_days':
                trsac[col] = self.today - trsac[col]
            elif colum_proccessed[col] == 'classify':
                trsac = self.column_classify(trsac, col, classify_dict[col])
            elif colum_proccessed[col] == 'one_hot_encode':
                trsac = self.one_hot_encode(trsac, col)
            else:
                pass
        
        trsac = trsac.fillna(0)
        trsac['CUST_ID'] = trsac['CUST_ID'].map(id2index)
        trsac = trsac.dropna(subset=['CUST_ID'])
        trsac['CUST_ID'] = trsac['CUST_ID'].astype(int)

        return trsac

    def column_classify(self, df, column, standard):
        if len(standard) == 1:
            df[column] = df[column].apply(lambda x: 1 if x in standard[0] else 0)
        else:
            for idx, std in enumerate(standard):
                df[f"{column}_class_{idx}"] = df[column].apply(lambda x: 1 if x in std else 0)
                df = df.drop(columns=[column])
        return df


    def one_hot_encode(self, df, column):
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)
        df = pd.concat([df, one_hot], axis=1)
        return df
    

    

def test():
    path = './comp_data/Train/'
    reader = file_reader(path)
    acc_info, label, id2index = reader.read_merge_info()
    print(acc_info.head())
    print(label.head())
    print("Account information shape:", acc_info.shape)
    print("Number of suspicious account:",label.sum())


if __name__ == "__main__":
    test()
