import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder




class train_file_reader:
    def __init__(self, path):
        self.pth = {
            'acc':path + '(Train)ACCTS_Data_202412.csv',
            'eccus':path + '(Train)ECCUS_Data_202412.csv',
            'id':path + '(Train)ID_Data_202412.csv',
            'trsac':path + '(Train)SAV_TXN_Data_202412.csv',
        }
        self.today = 18320
        self.read_account_info()
        self.read_transaction_info()


    def get_transaction_merge_into_account(self) -> pd.DataFrame:
        '''
            把交易的資料融合進入帳戶資料中，並附上該帳戶前5大交易的完整資訊（欄位展開）

            return:
                acc_info: 帳戶資料
        '''
        # Customizable
        merge_processed = {
            'TX_DATE': lambda x: (x.max() - x.min()) / len(x) if len(x) > 1 else 30,
        }
        fill_na = {
            'TX_DATE': 30,
            'TRN_COUNT': 0,
        }

        # --- 基本聚合 ---
        transaction_info = self.transac_info.sort_values(by=['ACCT_NBR', 'TX_DATE'])
        transaction_aggregated = transaction_info.groupby('ACCT_NBR').agg(merge_processed).reset_index(drop=True)
        transaction_aggregated['ACCT_NBR'] = transaction_info['ACCT_NBR'].unique()

        # --- ✨ 新增：每個帳戶的小額、中額、大額交易筆數 ✨ ---

        # 先定義小中大額的範圍（你可以調整）
        small_threshold = 500
        large_threshold = 100000
        
        # 幫每筆交易標小中大
        transaction_info['AMT_SIZE'] = pd.cut(
            transaction_info['TX_AMT'],
            bins=[1, small_threshold, large_threshold, np.inf],
            labels=['small', 'medium', 'large']
        )

        # 算每個帳戶小中大額交易的筆數
        amt_size_counts = transaction_info.pivot_table(
            index='ACCT_NBR',
            columns='AMT_SIZE',
            values='TX_AMT',
            aggfunc='count'
        ).fillna(0).reset_index()


        # 為了保險，確保三個欄位都有
        for size in ['small', 'medium', 'large']:
            if size not in amt_size_counts.columns:
                amt_size_counts[size] = 0
        
        amt_ratio_counts = amt_size_counts
        amt_size_counts = amt_size_counts.rename(columns={
            'small': 'SMALL_TRN_COUNT',
            'medium': 'MEDIUM_TRN_COUNT',
            'large': 'LARGE_TRN_COUNT'
        })
        
        amt_ratio_counts = amt_ratio_counts.rename(columns={
            'small': 'SMALL_TRN_RATIO',
            'medium': 'MEDIUM_TRN_RATIO',
            'large': 'LARGE_TRN_RATIO'
        })

        sum = amt_size_counts['SMALL_TRN_COUNT'] + amt_size_counts['MEDIUM_TRN_COUNT'] + amt_size_counts['LARGE_TRN_COUNT']
        
        amt_ratio_counts['SMALL_TRN_RATIO'] /= sum
        amt_ratio_counts['MEDIUM_TRN_RATIO'] /= sum
        amt_ratio_counts['LARGE_TRN_RATIO'] /= sum
        #print(amt_size_counts)

        # merge 回 transaction_aggregated
        transaction_aggregated = pd.merge(transaction_aggregated, amt_size_counts, on='ACCT_NBR', how='left')
        transaction_aggregated = pd.merge(transaction_aggregated, amt_ratio_counts, on='ACCT_NBR', how='left')

        # 填補空值
        transaction_aggregated[['SMALL_TRN_COUNT', 'MEDIUM_TRN_COUNT', 'LARGE_TRN_COUNT','SMALL_TRN_RATIO','MEDIUM_TRN_RATIO','LARGE_TRN_RATIO']] = transaction_aggregated[
            ['SMALL_TRN_COUNT', 'MEDIUM_TRN_COUNT', 'LARGE_TRN_COUNT','SMALL_TRN_RATIO','MEDIUM_TRN_RATIO','LARGE_TRN_RATIO']
        ].fillna(0)
        
        transaction_info = transaction_info.drop(columns=['AMT_SIZE'])

        remaining_columns = set(transaction_info.columns) - {'ACCT_NBR','AMT_SIZE'} - set(merge_processed.keys())

        for col in remaining_columns:
            transaction_aggregated[col] = transaction_info.groupby('ACCT_NBR')[col].mean().reset_index(drop=True)

        # Number of transactions
        transaction_aggregated['TRN_COUNT'] = transaction_info.groupby('ACCT_NBR').size().reset_index(drop=True)

        # --- ✨ 新增：取前5大交易，展開到欄位 ✨ ---
        transaction_info_sorted = transaction_info.sort_values(by=['ACCT_NBR', 'TX_AMT'], ascending=[True, False])

        top5 = transaction_info_sorted.groupby('ACCT_NBR').head(5)
        top5['TOP_N'] = top5.groupby('ACCT_NBR').cumcount() + 1

        le = LabelEncoder()
        object_columns = top5.select_dtypes(include=['object']).columns

        for col in object_columns:
            top5[col] = top5[col].fillna('UNKNOWN')  # Or any placeholder you prefer
        
        for col in object_columns:
            top5[col] = le.fit_transform(top5[col].astype(str))
        
        print("top5 中的欄位：", top5.columns)

        # 每筆交易攤平成多欄（例如 TX_DATE_1, TX_AMT_1, MCC_CD_1）
        top5_columns_to_expand = top5.columns.drop('ACCT_NBR')

        top5_list = []
        for n in range(1, 6):
            temp = top5[top5['TOP_N'] == n].copy()
            temp = temp.set_index('ACCT_NBR')
            temp = temp[top5_columns_to_expand]  # ⭐ 只選這兩個欄位
            temp.columns = [f"{col}_{n}" for col in temp.columns]
            top5_list.append(temp)


        # 把五組交易 merge 起來
        if top5_list:
            top5_merged = pd.concat(top5_list, axis=1)
            top5_merged = top5_merged.reset_index()
        else:
            top5_merged = pd.DataFrame()

        # --- 合併 ---
        acc_info = pd.merge(self.acc_info, transaction_aggregated, on='ACCT_NBR', how='left')
        acc_info = pd.merge(acc_info, top5_merged, on='ACCT_NBR', how='left')


        # Fill NAN
        acc_info = acc_info.fillna(fill_na)
        for col in remaining_columns:
            acc_info[col] = acc_info[col].fillna(0)

        for col in top5_merged.columns:
            #print(col)
            if col != 'ACCT_NBR':
                if acc_info[col].dtype == 'object':
                    acc_info[col] = le.fit_transform(acc_info[col].fillna('UNKNOWN').astype(str))
                else:
                    acc_info[col] = acc_info[col].fillna(0)

        acc_info = acc_info.drop(columns=['ACCT_NBR'])
        return acc_info, self.label

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
            'CNTY_CD':'classify'
        }
        # Customizable
        classify_dict = {
            'CNTY_CD': [[12]]
        }

        acc = pd.read_csv(self.pth['acc'])
        id = pd.read_csv(self.pth['id'])
        merged_data = pd.merge(acc, id, on='CUST_ID', how='left')
        merged_data = merged_data.drop(columns=['CUST_ID'], errors='ignore')

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

        merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].map(id2index)
        merged_data = merged_data.dropna(subset=['ACCT_NBR'])
        merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].astype(int)
        merged_data = merged_data.sort_values(by='ACCT_NBR') 

        # Label
        eccus = pd.read_csv(self.pth['eccus'])
        eccus = eccus.drop_duplicates(subset=['ACCT_NBR'])
        eccus['ACCT_NBR'] = eccus['ACCT_NBR'].map(id2index)
        label = merged_data['ACCT_NBR'].isin(eccus['ACCT_NBR']).astype(int).reset_index(drop=True)


        
        
        self.acc_info = merged_data
        self.label = label
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
            'CHANNEL_CODE':'classify',
            'TRN_CODE':'classify',
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
            'DAY_OF_WEEK': [['Saturday', 'Sunday']],
            'TRN_CODE': [[1, 4, 7, 8, 21, 29, 30, 24, 25, 26, 27, 28, 36, 37, 38],[2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 42, 44],[8, 9, 19, 52, 53, 54],[31, 32, 33, 34, 35, 39, 40, 45, 46, 47, 48, 49, 50],[6, 16, 41, 43, 51]],
            'CHANNEL_CODE': [[13,14,15,16,17,18],[5,6,9,12],[1,2,3,4,8,10],[7,11,19]]
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
        label = merged_data['ACCT_NBR'].map(self.label)

        return merged_data, label

    def get_info(self):
        return self.acc_info, self.label, self.transac_info

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
    reader = train_file_reader(path)
    print("Testing initialization:")
    print("Account info shape:", reader.acc_info.shape)
    print("Transaction info shape:", reader.transac_info.shape)
    print("Label shape:", reader.label.shape)
    print("Label sum:", reader.label.sum())

    print("Testing get_transaction_merge_into_account():")
    acc_info, label = reader.get_transaction_merge_into_account()
    print(acc_info.shape)
    print(label.shape)
    print(label.sum())
    print("Testing get_account_merge_into_transaction():")
    merged_data, label = reader.get_account_merge_into_transaction()
    print(merged_data.shape)
    print(label.shape)
    print(label.sum())



if __name__ == "__main__":
    test()
