import pandas as pd

# Read the CSV files
product_data = pd.read_csv("product_data.csv")
targets = pd.read_csv("targets.csv")

# Convert 'cod_modelo_color' to string for both DataFrames
product_data['cod_modelo_color'] = product_data['cod_modelo_color'].astype(str)
targets['cod_modelo_color'] = targets['cod_modelo_color'].astype(str)

# Select required columns from product_data
product_data_subset = product_data[[
    'cod_modelo_color', 'cod_color', 'des_sex', 'des_age', 'des_line', 
    'des_fabric', 'des_product_category', 'des_product_aggregated_family', 
    'des_product_family', 'des_product_type', 'des_filename', 'des_color'
]]

# Merge the two DataFrames on 'cod_modelo_color'
image_targets = pd.merge(product_data_subset, targets, on='cod_modelo_color', how='left')

# Save the merged DataFrame to a CSV file
image_targets.to_csv("images_targets2.csv", index=False)

print("Merged data has been saved to 'images_targets.csv'")
