import lab02_soil_analysis as hw
file_path = "soil_test.csv"
raw_data = hw.load_data(file_path)
clean_data = hw.clean_data(raw_data)
manipluatable_data = clean_data[:1]
for column in manipluatable_data:
    hw.compute_statistics(clean_data, column)

