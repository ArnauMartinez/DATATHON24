import pandas as pd

targets = pd.read_csv("targets.csv", sep=",")
product_data = pd.read_csv("product_data.csv", sep=",")
p_data = product_data[['cod_modelo_color', 'des_filename']]


image_targets = pd.concat(objs=[p_data, targets], axis=1)
image_targets = image_targets[1,:]
image_targets.to_csv("images_targets.csv", index=False)

