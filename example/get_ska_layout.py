import numpy as np

def get_latest_SKA_Low_layout(
                subarray_type='AA*',
                custom_stations=None,
                add_stations=None,
                exclude_stations=None,
                external_telescopes=None,
            ):
    try:
        from ska_ost_array_config.array_config import LowSubArray, filter_array_by_distance
    except:
        print(f'Install the official tool from SKAO provided at')
        print(f'https://gitlab.com/ska-telescope/ost/ska-ost-array-config')
        return None
    
    low_all = LowSubArray(
                subarray_type=subarray_type,
                custom_stations=custom_stations,
                add_stations=add_stations,
                exclude_stations=exclude_stations,
                external_telescopes=external_telescopes,
            )
    return low_all
    