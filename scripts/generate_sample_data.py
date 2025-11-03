import pandas as pd

def generate_data():
    # Carbon Emission Data
    carbon_emission_data = {
        'LE_ID': ['LE1', 'LE2'],
        'Financed_Emission': [100, 200],
    }
    carbon_emission_df = pd.DataFrame(carbon_emission_data)
    carbon_emission_df.to_csv('/Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_Training/data/scb-data/carbon_emission_data.csv', index=False)

    # Client Analytics Data
    client_analytics_data = {
        'LE_ID': ['LE1', 'LE2'],
        'Total_Revenue_YTD': [1000, 2000],
    }
    client_analytics_df = pd.DataFrame(client_analytics_data)
    client_analytics_df.to_csv('/Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_Training/data/scb-data/client_analytics_data.csv', index=False)

    # Transactions Data
    transactions_data = {
        'LE_ID': ['LE1', 'LE2'],
        'Amount': [50, 100],
    }
    transactions_df = pd.DataFrame(transactions_data)
    transactions_df.to_csv('/Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_Training/data/scb-data/transactions_data.csv', index=False)

if __name__ == '__main__':
    generate_data()
