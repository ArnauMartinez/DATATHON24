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
    writer.writerow(["cod_modelo_color", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142"])
    
    for cod_modelo_color, attributes in codes_attributes.items():
        all_values: list[int] = []
        
        for attribute_name in attributes_names:
            all_values.extend(attributes[attribute_name])
        
    
        writer.writerow([str(cod_modelo_color)] + all_values)