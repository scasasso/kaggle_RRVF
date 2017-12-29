
# String to append to output files
TAG = 'test'

# Features selected
# SEL_FEATURES = ['label', 'day_of_week', 'air_store_id', 'delta', 'reserved_visitors', 'genre', 'area',
#                 'res_1wk', 'res_2wk', 'res_1mo', 'res_2mo', 'res_3mo', 'res_6mo', 'res_1yr', 'visits_3mo_dow',
#                 'visits_6mo_dow', 'visits_1yr_dow', 'visits_1yr_holiday', 'n_open_days_1yr',
#                 'latitude', 'longitude']  # all

# SEL_FEATURES = ['day_of_week', 'delta', 'reserved_visitors', 'genre', 'area',
#                 'res_1wk', 'res_2wk', 'res_1mo', 'res_2mo', 'res_3mo', 'res_6mo', 'res_1yr', 'visits_3mo_dow',
#                 'visits_6mo_dow', 'visits_1yr_dow', 'visits_1yr_holiday', 'n_open_days_1yr']

# SEL_FEATURES = ['day_of_week', 'reserved_visitors', 'genre', 'delta',
#                 'res_3mo', 'res_6mo', 'res_1yr',
#                 'visits_3mo_dow', 'visits_6mo_dow', 'visits_1yr_dow']

# SEL_FEATURES = ['day_of_week', 'delta', 'reserved_visitors', 'genre',
#                 'res_1wk', 'res_2wk', 'res_1mo', 'res_2mo', 'res_3mo',
#                 'n_open_days_1yr']  # 0.555

SEL_FEATURES = ['day_of_week', 'reserved_visitors', 'genre']

# Target
TARGET_COL = 'visitors'
