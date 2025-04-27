import pandas as pd
import numpy as np


class test_file_reader:
    def __init__(self, path):
        self.pth = {
            'acc': path + '(Test)ACCTS_Data_202501.csv',
            'answersheet': path + '服務器繁忙，請稍後再試_test_result.csv',
            'id': path + '(Test)ID_Data_202501.csv',
            'trsac': path + '(Test)SAV_TXN_Data_202501.csv',
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

        try:
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
                    # For non-numeric columns, use first value
                    temp_agg = transaction_info.groupby('ACCT_NBR')[col].first().reset_index()
                
                transaction_aggregated = pd.merge(transaction_aggregated, temp_agg, on='ACCT_NBR', how='left')

            # Number of transactions
            trn_count = transaction_info.groupby('ACCT_NBR').size().reset_index(name='TRN_COUNT')
            transaction_aggregated = pd.merge(transaction_aggregated, trn_count, on='ACCT_NBR', how='left')

            # Merge with account info, preserving the original ACCT_NBR
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

            # Store original ACCT_NBR for mapping back later
            self.original_acct_nbr = acc_info['ACCT_NBR'].copy()
            
            # Make sure all columns are numeric for model training
            # First, separate the account number column
            acct_nbr = acc_info['ACCT_NBR'].copy()
            
            # Drop non-feature columns
            cols_to_drop = ['ORIGINAL_ACCT_NBR']
            X = acc_info.drop(columns=cols_to_drop, errors='ignore')
            
            # Convert any remaining non-numeric columns to numeric if possible
            for col in X.columns:
                if col != 'ACCT_NBR' and not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    except:
                        # If can't convert to numeric, drop the column
                        print(f"Dropped non-numeric column: {col}")
                        X = X.drop(columns=[col])
            
            return X
            
        except Exception as e:
            print(f"Error in transaction aggregation: {e}")
            # Return a basic structure if there's an error
            return self.acc_info

    def read_account_info(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        '''
            Read account information and merge with ID information.
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

        try:
            acc = pd.read_csv(self.pth['acc'])
            id_df = pd.read_csv(self.pth['id'])
            
            # Store original account numbers before any mapping
            self.original_account_numbers = acc['ACCT_NBR'].copy()
            
            merged_data = pd.merge(acc, id_df, on='CUST_ID', how='left')
            
            # Create id2index mapping using original account numbers
            all_ids = merged_data['ACCT_NBR'].unique()
            self.id2index = {cid: idx for idx, cid in enumerate(all_ids)}
            
            # Also create the reverse mapping for easy conversion back
            self.index2id = {idx: cid for cid, idx in self.id2index.items()}
            
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
                    
            # Store the original ACCT_NBR values before mapping
            merged_data['ORIGINAL_ACCT_NBR'] = merged_data['ACCT_NBR'].copy()
            
            # Apply the id2index mapping but keep the original ACCT_NBR for reference
            merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].map(self.id2index)
            
            # Keep only rows with valid ACCT_NBR mapping
            merged_data = merged_data.dropna(subset=['ACCT_NBR'])
            merged_data['ACCT_NBR'] = merged_data['ACCT_NBR'].astype(int)
            merged_data = merged_data.sort_values(by='ACCT_NBR')

            self.acc_info = merged_data
            
        except Exception as e:
            print(f"Error in account info processing: {e}")
            # Create a minimal DataFrame if there's an error
            self.acc_info = pd.DataFrame(columns=['ACCT_NBR', 'ORIGINAL_ACCT_NBR'])
            self.id2index = {}
            self.index2id = {}
            self.original_account_numbers = pd.Series([])

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
            
            # Store original account numbers before mapping
            trsac['ORIGINAL_ACCT_NBR'] = trsac['ACCT_NBR'].copy()
            
            # Process non-numeric columns to prevent issues later
            for col in trsac.columns:
                if col not in ['ACCT_NBR', 'ORIGINAL_ACCT_NBR']:
                    if not pd.api.types.is_numeric_dtype(trsac[col]):
                        try:
                            trsac[col] = pd.to_numeric(trsac[col], errors='coerce')
                        except:
                            # Keep as is if conversion fails
                            pass
            
            # Fill NaN values after type conversion
            trsac = trsac.fillna(0)
            
            # Map account numbers to indices
            trsac['ACCT_NBR'] = trsac['ACCT_NBR'].map(self.id2index)
            
            # Keep only rows with valid mapping
            trsac = trsac.dropna(subset=['ACCT_NBR'])
            trsac['ACCT_NBR'] = trsac['ACCT_NBR'].astype(int)

            self.transac_info = trsac
            
        except Exception as e:
            print(f"Error in transaction info processing: {e}")
            # Create an empty DataFrame if there's an error
            self.transac_info = pd.DataFrame(columns=['ACCT_NBR', 'TX_DATE'])
            self.original_txn_acct_nbr = pd.Series([])

    def get_account_merge_into_transaction(self) -> tuple[pd.DataFrame, pd.Series]:
        '''
            把帳戶的資料融合進入交易資料中

            return:
                merged_data: 融合後的交易資料
                label: 標籤
        '''
        try:
            merged_data = pd.merge(self.transac_info, self.acc_info, on='ACCT_NBR', how='left')
            return merged_data
        except Exception as e:
            print(f"Error in merging account into transaction: {e}")
            return pd.DataFrame(columns=['ACCT_NBR'])

    def write_answer(self, predictions):
        '''
            Write the answer to the file
        '''
        try:
            ans = pd.read_csv(self.pth['answersheet'])
            
            # Create a mapping from original account numbers to predictions
            pred_dict = {}
            for idx, pred in enumerate(predictions):
                acct_idx = idx
                if acct_idx in self.index2id:
                    original_acct = self.index2id[acct_idx]
                    pred_dict[original_acct] = pred
            
            # Map predictions to answer sheet
            ans['Y'] = ans['ACCT_NBR'].map(pred_dict).apply(lambda x: "1" if x == 1 else "")
            
            duplicate_count = ans['ACCT_NBR'].duplicated(keep=False).sum()
            print(f"Number of duplicate 'ACCT_NBR': {duplicate_count}")
            
            ans.to_csv(self.pth['answersheet'], index=False)
            
        except Exception as e:
            print(f"Error writing answer: {e}")
        
    def clean_answer(self):
        '''
            Clean the answer file
        '''
        try:
            ans = pd.read_csv(self.pth['answersheet'])
            ans['Y'] = ""
            ans.to_csv(self.pth['answersheet'], index=False)
        except Exception as e:
            print(f"Error cleaning answer: {e}")

    def get_info(self):
        return self.acc_info, self.transac_info

    def column_classify(self, df, column, standard):
        if column not in df.columns:
            return df
            
        if len(standard) == 1:
            df[column] = df[column].apply(lambda x: 1 if x in standard[0] else 0)
        else:
            for idx, std in enumerate(standard):
                df[f"{column}_class_{idx}"] = df[column].apply(lambda x: 1 if x in std else 0)
                df = df.drop(columns=[column], errors='ignore')
        return df

    def one_hot_encode(self, df, column):
        if column not in df.columns:
            return df
            
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(columns=[column], errors='ignore')
        df = pd.concat([df, one_hot], axis=1)
        return df
    
    def get_original_acct_mapping(self):
        """
        Return mapping from internal indices to original account numbers
        """
        return self.index2id