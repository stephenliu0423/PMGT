

def get_rec_config(data_type):
    print(data_type)
    # num_items keep same as pre-training graph
    # some item no appear in downstream task don't need gradient update
    # but still exist in embedding matrix
    if data_type == 'video':
        rec_config = {
            'num_items': 7252,
            'num_users': 27988,
            'num_interactions': 98278
        }
    elif data_type == 'toys':
        rec_config = {
            'num_items': 6451,
            'num_users': 118153,
            'num_interactions': 294507
        }
    elif data_type == 'tools':
        rec_config = {
            'num_items': 5982,
            'num_users': 164717,
            'num_interactions': 431455
        }
    elif data_type == 'movie_lens':
        rec_config = {
            'num_items': 000,
            'num_users': 000,
            'num_interactions': 000
        }
    else:
        raise RuntimeError('please input right data type')
    return rec_config

