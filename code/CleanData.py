class CleanData:
    with open('../data/slump_test.data.txt', 'r') as slump_data:
        first_line = True
        # initialize a list that will contain the data of each row with
        # the index being the row number, works like a dictionary with key as
        # the row number
        data_as_list = [0] * 103
        for line in slump_data:
            # We do not need the first line since its the column names
            # skip the first line before inserting to data list
            if first_line:
                cols = line.split(',')
                first_line = False
            else:
                data_as_list[int(line.split(',')[0]) - 1] = line.strip().split(',')[0:8] + line.strip().split(',')[9:10]
    # (x) variables by index are:
    # {number, cement, slag, fly ash, water, superplasticizer,
    # coarse aggregate, fine aggregate}
    # (y) is the slump in cm

    def get_data(self):
        return self.data_as_list
