import get_time_range

def get_time_index(time_range):

    start_date_target, end_date_target = \
        time_range['target']['start_date'], time_range['target']['end_date'] 

    time_idx_range = {}
    print('time_range keys', time_range.keys())

    for i_model in time_range.keys():
        time_idx_range[i_model] = {}
        print('i_model:', i_model)
        if i_model != 'target':
            for i_var in time_range[i_model].keys():
                start_idx, end_idx = [], []
                start_date_all, end_date_all, step_hours = \
                    time_range[i_model][i_var]['start_date'], time_range[i_model][i_var]['end_date'], int(time_range[i_model][i_var]['step_hours']) 

                for i in range(len(start_date_all)):
                    all_times  = get_time_range.generate_time_series(start_date_all[i], end_date_all[i], step_hours)
                    start_idx_i,  end_idx_i  = get_time_range.get_time_indices(all_times, start_date_target[i], end_date_target[i])
                    start_idx.append(start_idx_i)
                    end_idx.append(end_idx_i)
                    print(f"Start time all: {all_times[start_idx_i]}, End time all: {all_times[end_idx_i]}")
                time_idx_range[i_model][i_var] = {'start_idx': start_idx, 'end_idx': end_idx}
                print(f"Start index target: {start_idx}, End index target: {end_idx}")

    print(time_idx_range)
    return time_idx_range
