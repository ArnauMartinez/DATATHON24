import csv

codes_attributes:dict[str, dict[str, list[int]]] = {}

attributes_names = ["cane_height_type", "closure_placement", "heel_shape_type", "knit_structure", "length_type", 
                   "neck_lapel_type", "silhouette_type", "sleeve_length_type", "toecap_type", "waist_type", "woven_structure"]

attributes_types = [3, 6, 13, 5, 13, 35, 50, 6, 4, 4, 4]

# Read the CSV
with open("/Users/laura/DATATHON24/DATATHON24/data/archive/attribute_data.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        cod_modelo_color = row["cod_modelo_color"]
        attribute_name = row["attribute_name"]
        cod_value = int(row["cod_value"])
        
        # If not exists, creates a new entrance
        if cod_modelo_color not in codes_attributes:
            codes_attributes[cod_modelo_color] = {attributes_names[i]: [0 for _ in range(attributes_types[i])] for i in range(11)}
        
        if attribute_name == "knit_structure":
            codes_attributes[cod_modelo_color][attribute_name][cod_value] = 1
        else:
            codes_attributes[cod_modelo_color][attribute_name][cod_value-1] = 1

with open("targets.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    for cod_modelo_color, attributes in codes_attributes.items():
        all_values: list[int] = []
        
        for attribute_name in attributes_names:
            all_values.extend(attributes[attribute_name])
        
    
        writer.writerow([cod_modelo_color] + all_values)