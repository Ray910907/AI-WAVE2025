import pandas as pd
import numpy as np


class test_file_reader:
    def __init__(self, path):
        self.pth = {
            'acc':path + '(Test)ACCTS_Data_202501.csv',
            'answersheet':path + '(Test)ANS_參賽者_202501.csv',
            'id':path + '(Test)ID_Data_202501.csv',
            'trsac':path + '(Test)SAV_TXN_Data_202501.csv',
        }
        self.today = 18320
        self.read_account_info()
        self.read_transaction_info()

    def get_transaction_merge_into_account(self) -> pd.DataFrame:
        '''
            把交易的資料融合進入帳戶資料中

            return:
                acc_info: 帳戶資料
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


        # Process transaction information
        transaction_info = self.transac_info.sort_values(by=['ACCT_NBR', 'TX_DATE'])
        transaction_aggregated = transaction_info.groupby('ACCT_NBR').agg(merge_processed).reset_index(drop=True)
        transaction_aggregated['ACCT_NBR'] = transaction_info['ACCT_NBR'].unique()
        remaining_columns = set(transaction_info.columns) - {'ACCT_NBR'} - set(merge_processed.keys())
        for col in remaining_columns:
            transaction_aggregated[col] = transaction_info.groupby('ACCT_NBR')[col].mean().reset_index(drop=True)

        # Number of transactions
        transaction_aggregated['TRN_COUNT'] = transaction_info.groupby('ACCT_NBR').size().reset_index(drop=True)

        # Merge
        acc_info = pd.merge(self.acc_info, transaction_aggregated, on='ACCT_NBR', how='left')


        #fill NAN
        acc_info = acc_info.fillna(fill_na)
        for col in remaining_columns:
            acc_info[col] = acc_info[col].fillna(0)

        return acc_info

    def read_account_info(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        '''
            Read account information and merge with ID information.
            Label is 1 if the account is in eccus, 0 otherwise.
        '''
        # Customizable
        colum_proccessed = {
            'CANCEL_NO_CONTACT':'none',
            'IS_DIGITAL':'none',
            'ACCT_OPEN_DT':'turn_to_days',
            'AUM_AMT':'none',
            'DATE_OF_BIRTH':'none',
            'YEARLYINCOMELEVEL':'none',
            'CNTY_CD':'one_hot_encode'
        }
        # Customizable
        classify_dict = {
            'CNTY_CD': [['CN', 'TW', 'HK', 'MO', 'SG', 'JP', 'KR']]
        }

        acc = pd.read_csv(self.pth['acc'])
        id = pd.read_csv(self.pth['id'])
        merged_data = pd.merge(acc, id, on='CUST_ID', how='left')
        merged_data = merged_data.drop(columns=['CUST_ID'])

        for col in colum_proccessed.keys():
            if colum_proccessed[col] == 'drop':
                merged_data = merged_data.drop(columns=[col])
            elif colum_proccessed[col] == 'turn_to_days':
                merged_data[col] = self.today - merged_data[col]
            elif colum_proccessed[col] == 'one_hot_encode':
                merged_data = self.one_hot_encode(merged_data, col)
            elif colum_proccessed[col] == 'classify':
                merged_data = self.column_classify(merged_data, col, classify_dict[col])
            else:
                pass
        
        # id2index
        all_ids = merged_data['ACCT_NBR'].unique()
        id2index = {cid: idx for idx, cid in enumerate(all_ids)}
    
        
        # Customizable fill values for each column
        fill_na_values = {
            'AUM_AMT': 50000,
            'IS_DIGITAL': 0,
            'DATE_OF_BIRTH': 20,
            'YEARLYINCOMELEVEL': 0,
            'CNTY_CD': 12,
            'CANCEL_NO_CONTACT':0,
            'IS_DIGITAL': 0,
            'ACCT_OPEN_DT': 100,
            
        }

        # Fill NaN values based on the specified dictionary
        for col, fill_value in fill_na_values.items():
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna(fill_value)
        # mapping
        merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].map(id2index)
        merged_data = merged_data.dropna(subset=['ACCT_NBR'])
        merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].astype(int)
        merged_data = merged_data.sort_values(by='ACCT_NBR')


        
        
        self.acc_info = merged_data
        self.id2index = id2index
        

    def read_transaction_info(self) -> pd.DataFrame:
        '''
            Read transaction information and process it.
        '''
        # Customizable
        colum_proccessed = {
            'CUST_ID':'drop',
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
        trsac = trsac.drop_duplicates()
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
        trsac['ACCT_NBR'] = trsac['ACCT_NBR'].map(self.id2index)
        trsac = trsac.dropna(subset=['ACCT_NBR'])
        trsac['ACCT_NBR'] = trsac['ACCT_NBR'].astype(int)

        self.transac_info = trsac

    def get_account_merge_into_transaction(self) -> tuple[pd.DataFrame, pd.Series]:
        '''
            把帳戶的資料融合進入交易資料中

            return:
                merged_data: 融合後的交易資料
                label: 標籤
        '''

        merged_data = pd.merge(self.transac_info, self.acc_info, on='ACCT_NBR', how='left')

        return merged_data
    def write_answer(self, predictions):
        '''
            Write the answer to the file
        '''
        ans = pd.read_csv(self.pth['answersheet'])
        print(ans.shape)
        ans['Y'] = ans['ACCT_NBR'].map(self.id2index).map(predictions).apply(lambda x: "1" if x == 1 else "")
        duplicate_count = ans['ACCT_NBR'].duplicated(keep=False).sum()
        print(f"Number of duplicate 'ACCT_NBR': {duplicate_count}")
        ans.to_csv(self.pth['answersheet'], index=False)
        
    
    def clean_answer(self):
        '''
            Clean the answer file
        '''
        ans = pd.read_csv(self.pth['answersheet'])
        ans['Y'] = ""
        ans.to_csv(self.pth['answersheet'], index=False)

    def get_info(self):
        return self.acc_info, self.transac_info

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
    path = './comp_data/Test/'
    reader = test_file_reader(path)
    reader.clean_answer()

    print("Testing...")
    acc_info, transac_info = reader.get_info()
    print("Account info shape:", acc_info.shape)
    print(acc_info.head())
    print("Transaction info shape:", transac_info.shape)
    print(transac_info.head())

    print("Testing get_transaction_merge_into_account():")
    acc_info = reader.get_transaction_merge_into_account()
    print(acc_info.shape)
    print(acc_info.head())

    print("Testing get_account_merge_into_transaction():")
    merged_data = reader.get_account_merge_into_transaction()
    print(merged_data.shape)

    print("Testing write_answer():")
    predictions = np.random.randint(0, 2, size=acc_info.shape[0])
    predictions = pd.Series(predictions, index=acc_info.index)
    print(predictions.head())
    reader.write_answer(predictions)




if __name__ == "__main__":
    test()
