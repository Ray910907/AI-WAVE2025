import pandas as pd
import numpy as np


class train_file_reader:
    def __init__(self, path):
        self.pth = {
            'acc': path + '(Train)ACCTS_Data_202412.csv',
            'eccus': path + '(Train)ECCUS_Data_202412.csv',
            'id': path + '(Train)ID_Data_202412.csv',
            'trsac': path + '(Train)SAV_TXN_Data_202412.csv',
        }
        self.today = 18290
        self.read_account_info()
        self.read_transaction_info()

    def get_transaction_merge_into_account(self) -> tuple[pd.DataFrame, pd.Series]:
        '''
            把交易的資料融合進入帳戶資料中

            return:
                acc_info: 帳戶資料
                label: 標籤
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
        
        # For columns with custom aggregation
        transaction_aggregated = transaction_info.groupby('ACCT_NBR').agg(merge_processed).reset_index()
        
        # Handle remaining columns for aggregation
        remaining_columns = set(transaction_info.columns) - {'ACCT_NBR'} - set(merge_processed.keys())
        
        for col in remaining_columns:
            # Check if column is numeric before taking mean
            is_numeric = pd.api.types.is_numeric_dtype(transaction_info[col])
            
            if is_numeric:
                # For numeric columns, use mean
                temp_agg = transaction_info.groupby('ACCT_NBR')[col].mean().reset_index()
            else:
                # For non-numeric columns, use first value or most common
                temp_agg = transaction_info.groupby('ACCT_NBR')[col].first().reset_index()
            
            transaction_aggregated = pd.merge(transaction_aggregated, temp_agg, on='ACCT_NBR', how='left')

        # Number of transactions
        trn_count = transaction_info.groupby('ACCT_NBR').size().reset_index(name='TRN_COUNT')
        transaction_aggregated = pd.merge(transaction_aggregated, trn_count, on='ACCT_NBR', how='left')

        # Merge with account info
        acc_info = pd.merge(self.acc_info, transaction_aggregated, on='ACCT_NBR', how='left')

        # Fill NaN values
        acc_info = acc_info.fillna(fill_na)
        for col in remaining_columns:
            if col in acc_info.columns:
                is_numeric = pd.api.types.is_numeric_dtype(acc_info[col])
                if is_numeric:
                    acc_info[col] = acc_info[col].fillna(0)
                else:
                    acc_info[col] = acc_info[col].fillna('')
        
        # Keep a copy of the original account mapping for later reference
        self.feature_df = acc_info.copy()
        
        # Make sure all columns are numeric for model training
        # Drop non-numeric columns and non-feature columns
        cols_to_drop = ['ACCT_NBR', 'ORIGINAL_ACCT_NBR']
        X = acc_info.drop(columns=cols_to_drop, errors='ignore')
        
        # Convert any remaining non-numeric columns to numeric if possible
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                except:
                    # If can't convert to numeric, drop the column
                    X = X.drop(columns=[col])
                    print(f"Dropped non-numeric column: {col}")

        return X, self.label

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
        id_df = pd.read_csv(self.pth['id'])
        
        # Store original account numbers
        self.original_account_numbers = acc['ACCT_NBR'].copy()
        
        merged_data = pd.merge(acc, id_df, on='CUST_ID', how='left')
        merged_data = merged_data.drop(columns=['CUST_ID'], errors='ignore')

        # Process columns
        for col in colum_proccessed.keys():
            if col in merged_data.columns:  # Make sure column exists
                if colum_proccessed[col] == 'drop':
                    merged_data = merged_data.drop(columns=[col], errors='ignore')
                elif colum_proccessed[col] == 'turn_to_days':
                    merged_data[col] = self.today - merged_data[col]
                elif colum_proccessed[col] == 'one_hot_encode':
                    merged_data = self.one_hot_encode(merged_data, col)
                elif colum_proccessed[col] == 'classify':
                    merged_data = self.column_classify(merged_data, col, classify_dict[col])
        
        # Create ID to index mapping
        all_ids = merged_data['ACCT_NBR'].unique()
        self.id2index = {cid: idx for idx, cid in enumerate(all_ids)}
        
        # Also create the reverse mapping
        self.index2id = {idx: cid for cid, idx in self.id2index.items()}
    
        # Customizable fill values for each column
        fill_na_values = {
            'AUM_AMT': 50000,
            'IS_DIGITAL': 0,
            'DATE_OF_BIRTH': 20,
            'YEARLYINCOMELEVEL': 0,
            'CNTY_CD': 12,
            'CANCEL_NO_CONTACT': 0,
            'ACCT_OPEN_DT': 100,
        }

        # Fill NaN values based on the specified dictionary
        for col, fill_value in fill_na_values.items():
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna(fill_value)

        # Store original account numbers before mapping
        merged_data['ORIGINAL_ACCT_NBR'] = merged_data['ACCT_NBR'].copy()
        
        # Map account numbers to indices
        merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].map(self.id2index)
        merged_data = merged_data.dropna(subset=['ACCT_NBR'])
        merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].astype(int)
        merged_data = merged_data.sort_values(by='ACCT_NBR') 

        # Create label vector
        eccus = pd.read_csv(self.pth['eccus'])
        eccus = eccus.drop_duplicates(subset=['ACCT_NBR'])
        
        # First map the ECCUS accounts to their indices
        eccus['idx'] = eccus['ACCT_NBR'].map(self.id2index)
        eccus = eccus.dropna(subset=['idx'])
        
        # Create label series (1 for accounts in ECCUS, 0 otherwise)
        label = merged_data['ACCT_NBR'].isin(eccus['idx'].astype(int)).astype(int)
        
        self.acc_info = merged_data
        self.label = label
        

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

        try:
            trsac = pd.read_csv(self.pth['trsac'])
            trsac = trsac.drop_duplicates()
            
            # Store original transaction account numbers
            self.original_txn_acct_nbr = trsac['ACCT_NBR'].copy()
            
            for col in colum_proccessed.keys():
                if col in trsac.columns:  # Make sure column exists
                    if colum_proccessed[col] == 'drop':
                        trsac = trsac.drop(columns=[col], errors='ignore')
                    elif colum_proccessed[col] == 'turn_to_days':
                        trsac[col] = self.today - trsac[col]
                    elif colum_proccessed[col] == 'classify':
                        trsac = self.column_classify(trsac, col, classify_dict[col])
                    elif colum_proccessed[col] == 'one_hot_encode':
                        trsac = self.one_hot_encode(trsac, col)
            
            # Fill NaN values
            trsac = trsac.fillna(0)
            
            # Store original account numbers before mapping
            trsac['ORIGINAL_ACCT_NBR'] = trsac['ACCT_NBR'].copy()
            
            # Make sure we're only dealing with numeric data for certain columns
            for col in trsac.columns:
                if col not in ['ACCT_NBR', 'ORIGINAL_ACCT_NBR', 'TX_DATE']:
                    try:
                        if not pd.api.types.is_numeric_dtype(trsac[col]):
                            trsac[col] = pd.to_numeric(trsac[col], errors='coerce').fillna(0)
                    except:
                        # If conversion fails, just fill with 0
                        trsac[col] = 0
            
            # Map account numbers to indices
            trsac['ACCT_NBR'] = trsac['ACCT_NBR'].map(self.id2index)
            
            # Keep only rows with valid mapping
            trsac = trsac.dropna(subset=['ACCT_NBR'])
            trsac['ACCT_NBR'] = trsac['ACCT_NBR'].astype(int)
            
            self.transac_info = trsac
            
        except Exception as e:
            print(f"Error processing transaction data: {e}")
            # Create empty DataFrame with required columns if there's an error
            self.transac_info = pd.DataFrame(columns=['ACCT_NBR', 'TX_DATE'])

    def get_account_merge_into_transaction(self) -> tuple[pd.DataFrame, pd.Series]:
        '''
            把帳戶的資料融合進入交易資料中

            return:
                merged_data: 融合後的交易資料
                label: 標籤
        '''
        merged_data = pd.merge(self.transac_info, self.acc_info, on='ACCT_NBR', how='left')
        
        # Create label for each transaction based on account
        label = merged_data['ACCT_NBR'].map(
            dict(zip(self.acc_info['ACCT_NBR'], self.label))
        ).fillna(0).astype(int)

        return merged_data, label

    def get_info(self):
        return self.acc_info, self.label, self.transac_info

    def column_classify(self, df, column, standard):
        if column not in df.columns:
            return df
            
        if len(standard) == 1:
            df[column] = df[column].apply(lambda x: 1 if x in standard[0] else 0)
        else:
            for idx, std in enumerate(standard):
                df[f"{column}_class_{idx}"] = df[column].apply(lambda x: 1 if x in std else 0)
                df = df.drop(columns=[column])
        return df

    def one_hot_encode(self, df, column):
        if column not in df.columns:
            return df
            
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)
        df = pd.concat([df, one_hot], axis=1)
        return df
    
    def get_original_acct_mapping(self):
        """
        Return mapping from internal indices to original account numbers
        """
        return self.index2id