def in_list_feature(x_train):
    days_months = ['Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday']
    days_months.extend(
        ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
         'December'])
    return [word in days_months for word in x_train]


def load_raw_dataset(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    dataset = [line.strip().split(sep=" ") for line in lines if not (line in ['\n', '. . O\n'])]

    x = [item[0] for item in dataset]
    feature = [item[1] for item in dataset]
    y = [[item[2]] for item in dataset]

    capital_feature = [word[0].isupper() for word in x]
    # abc_feature = [word[0].isalpha() for word in x] # didn't help our model
    # day_feature = InListFeature(x) # didn't help our model
    feature = [list(x) for x in zip(x, feature, capital_feature)]  # , abc_feature, day_feature)]
    return x, feature, y
