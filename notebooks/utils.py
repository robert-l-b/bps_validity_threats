import pandas as pd
import os

from log_distance_measures.config import EventLogIDs


log_ids = EventLogIDs(
    case="case:concept:name",
    activity="concept:name",
    start_time="start_timestamp",
    end_time="end_timestamp",
    resource="org:resource"
)

def group_lifecycle(log):


    # Step 1: Create a shifted version of the dataframe to compare consecutive rows
    log_shifted = log.shift(-1)

    # Step 2: Identify pairs where the first row has 'start' and the second has 'complete' in "lifecycle:transition"
    mask = (log["lifecycle:transition"] == "start") & (log_shifted["lifecycle:transition"] == "complete")

    # Step 3: Ensure all columns except "lifecycle:transition" and "time:timestamp" are the same for the pair
    columns_to_compare = log.columns.difference(["lifecycle:transition", "time:timestamp"])
    # same_values_mask = (log[columns_to_compare] == log_shifted[columns_to_compare]).all(axis=1)
    same_values_mask = (log[columns_to_compare].eq(log_shifted[columns_to_compare]) | 
                    (log[columns_to_compare].isna() & log_shifted[columns_to_compare].isna())).all(axis=1)


    # Step 4: Combine the two conditions (lifecycle:transition and matching values in other columns)
    valid_pairs_mask = mask & same_values_mask

    # Step 5: Create a new column "event_id" and assign unique IDs to the valid pairs
    log["event_id"] = None  # Initialize with None

    # Assign unique IDs to each valid pair
    event_id_counter = 1
    for idx in log[valid_pairs_mask].index:
        log.at[idx, "event_id"] = event_id_counter  # Assign to "start" row
        log.at[idx + 1, "event_id"] = event_id_counter  # Assign to "complete" row
        event_id_counter += 1

    # The dataframe 'log' now has a unique 'event_id' for each valid consecutive "start" and "complete

    return log


def convert_timestamps(log):

    log = group_lifecycle(log)

    # slit logs in start and complete
    log_start = log[log['lifecycle:transition'] == 'start']
    log_compl = log[log['lifecycle:transition'] == 'complete']


    # rename and drop column
    log_start = log_start.rename(columns = {'time:timestamp': 'start_timestamp'})
    log_compl = log_compl.rename(columns = {'time:timestamp': 'end_timestamp'})

    # filter for only necessary columns
    key_cols = ['case:concept:name', 'concept:name', 'event_id']
    filter_cols = key_cols.copy()
    filter_cols.append('start_timestamp')
    log_start = log_start[filter_cols]
    
    log_merged = log_compl.merge(log_start, left_on=key_cols, right_on=key_cols, how='left')

    return log_merged



def regular_cut(df, log_ids, criterium_value, cut_criterium='timestamp', one_timestamp=False, key_timestamp=None):
    # Group by 'case_id' and calculate the minimum 'start_timestamp' for each group
    grouped = df.groupby(log_ids.case).agg(min_start_timestamp=(key_timestamp, 'min')).reset_index()
    
    if cut_criterium == 'nr_cases':
        # Calculate the number of cases as a percentage of the total cases
        total_cases = grouped[log_ids.case].nunique()
        criterium_count = int(total_cases * criterium_value / 100)

        # Sort the groups based on the minimum 'start_timestamp' and select top cases
        sorted_case_ids = grouped.sort_values('min_start_timestamp')[log_ids.case]
        selected_case_ids = sorted_case_ids.head(criterium_count)

    elif cut_criterium == 'timestamp':
        # Filter the groups based on the cutoff timestamp
        selected_case_ids = grouped[grouped['min_start_timestamp'] <= criterium_value][log_ids.case]
    
    else:
        raise ValueError('cut_criterium incorrect')

    # Filter the original DataFrame based on the selected 'case_id's
    filtered_df = df[df[log_ids.case].isin(selected_case_ids)]
    remaining_df = df[~df[log_ids.case].isin(selected_case_ids)]

    # Sort both DataFrames by 'case_id' first and then by 'start_timestamp'
    filtered_df = filtered_df.sort_values(by=[log_ids.case, key_timestamp])
    remaining_df = remaining_df.sort_values(by=[log_ids.case, key_timestamp])

    return filtered_df, remaining_df


def intermediate_cut(df, log_ids, criterium_value, cut_criterium='timestamp', one_timestamp=False, key_timestamp=None):
    # Validate the cut_criterium
    if cut_criterium not in ['nr_events', 'nr_cases', 'timestamp']:
        raise ValueError("cut_criterium must be either 'nr_events' or 'timestamp")

    # Determine timestamps to use based on `one_timestamp`
    if one_timestamp:
        grouped = df.groupby(log_ids.case).agg(
            min_start_timestamp=(key_timestamp, 'min'),
            max_start_timestamp=(key_timestamp, 'max')
        ).reset_index()
    else:
        grouped = df.groupby(log_ids.case).agg(
            min_start_timestamp=(log_ids.start_time, 'min'),
            max_start_timestamp=(log_ids.end_time, 'max')
        ).reset_index()

    if cut_criterium == 'nr_events':
        # Calculate the number of events as a percentage of total events
        total_events = len(df)
        criterium_count = int(total_events * criterium_value / 100)

        # Sort and select top events
        sorted_case_ids = grouped.sort_values('min_start_timestamp')[log_ids.case]
        selected_case_ids = sorted_case_ids.head(criterium_count)

        # Determine cutoff timestamp
        selected_timestamps = grouped[grouped[log_ids.case].isin(selected_case_ids)]
        cutoff_timestamp = selected_timestamps['max_start_timestamp'].max() + pd.Timedelta(seconds=1)

    elif cut_criterium == 'timestamp':
        cutoff_timestamp = criterium_value

    elif cut_criterium == 'nr_cases':
        # Calculate number of cases as a percentage of total cases
        total_cases = grouped[log_ids.case].nunique()
        criterium_count = int(total_cases * criterium_value / 100)

        # Sort and select top cases
        sorted_case_ids = grouped.sort_values('min_start_timestamp')[log_ids.case]
        selected_case_ids = sorted_case_ids.head(criterium_count)

        # Determine cutoff timestamp
        selected_timestamps = grouped[grouped[log_ids.case].isin(selected_case_ids)]
        cutoff_timestamp = selected_timestamps['min_start_timestamp'].max() + pd.Timedelta(seconds=1)

    # Split DataFrame based on cutoff timestamp
    prior_df = df[df[key_timestamp] < cutoff_timestamp]
    after_df = df[df[key_timestamp] >= cutoff_timestamp]

    # Identify and remove cases that span across the cutoff timestamp
    exclude_cases = grouped[(grouped['min_start_timestamp'] <= cutoff_timestamp) &
                            (grouped['max_start_timestamp'] >= cutoff_timestamp)][log_ids.case]
    prior_df_filtered = prior_df[~prior_df[log_ids.case].isin(exclude_cases)]
    after_df_filtered = after_df[~after_df[log_ids.case].isin(exclude_cases)]

    prior_df_filtered = prior_df_filtered.sort_values(by=[log_ids.case, key_timestamp])
    after_df_filtered = after_df_filtered.sort_values(by=[log_ids.case, key_timestamp])

    return prior_df_filtered, after_df_filtered



def strict_cut(df, log_ids, criterium_value, cut_criterium='timestamp', one_timestamp=False, key_timestamp=None):
    if cut_criterium == 'nr_events':
        # Calculate the number of events as a percentage of total events
        total_events = len(df)
        criterium_count = int(total_events * criterium_value / 100)

        # Sort and select top events
        sorted_df = df.sort_values(by=key_timestamp)
        filtered_df = sorted_df.head(criterium_count)

    elif cut_criterium == 'timestamp':
        # Filter rows based on timestamp
        filtered_df = df[df[key_timestamp] < criterium_value]

    elif cut_criterium == 'nr_cases':
        # Calculate the number of cases based on the specified percentage
        total_cases = df[log_ids.case].nunique()
        criterium_count = int(total_cases * criterium_value / 100)

        # Group by case and find the minimum timestamp for each case
        grouped = df.groupby(log_ids.case).agg(min_start_timestamp=(key_timestamp, 'min')).reset_index()

        # Sort cases by their earliest timestamp and get the cutoff timestamp
        sorted_cases = grouped.sort_values('min_start_timestamp')
        cutoff_case_ids = sorted_cases.head(criterium_count)[log_ids.case]
        cutoff_timestamp = grouped[grouped[log_ids.case].isin(cutoff_case_ids)]['min_start_timestamp'].max()

        # Divide events by the calculated cutoff timestamp
        filtered_df = df[df[key_timestamp] < cutoff_timestamp]
    
    else:
        raise ValueError('cut_criterium incorrect')

    # Remaining DataFrame consists of events after the cutoff timestamp
    remaining_df = df[~df.index.isin(filtered_df.index)]

    return filtered_df, remaining_df



# ######################


def extract_start_in_time(df, log_ids, start_time, end_time, key_timestamp=None):
    """
    Extracts rows from a DataFrame that fall within the specified start and end timestamps.

    Parameters:
    - df: The input DataFrame to filter.
    - log_ids: An object containing log identifiers, specifically the case identifier and timestamp column.
    - start_time: The starting timestamp for filtering.
    - end_time: The ending timestamp for filtering.
    - key_timestamp: The column name for the timestamp, defaults to log_ids.start_time if not provided.

    Returns:
    - filtered_df: A DataFrame containing cases that start in the timespan.
    """

    # Set the key_timestamp to the provided value or use log_ids.start_time
    if key_timestamp is None:
        key_timestamp = log_ids.start_time

    # Use regular_cut to get the filtered DataFrame based on the start_time
    _, filtered_start_df = regular_cut(df, log_ids, start_time, cut_criterium='timestamp', key_timestamp=key_timestamp)

    # Use regular_cut to get the filtered DataFrame based on the end_time
    extracted_df, _ = regular_cut(filtered_start_df, log_ids, end_time, cut_criterium='timestamp', key_timestamp=key_timestamp)

    # # Get the intersection of the two filtered DataFrames to obtain the final filtered DataFrame
    # filtered_df = filtered_start_df[filtered_start_df[key_timestamp].between(start_time, end_time)]

    # # Sort the final filtered DataFrame by 'case_id' first and then by the timestamp
    # filtered_df = filtered_df.sort_values(by=[log_ids.case, key_timestamp])

    return extracted_df

def extract_contained_in_time(df, log_ids, start_time, end_time, key_timestamp=None, one_timestamp=False):
    """
    Extracts rows from a DataFrame that fall within the specified start and end timestamps.

    Parameters:
    - df: The input DataFrame to filter.
    - log_ids: An object containing log identifiers, specifically the case identifier and timestamp column.
    - start_time: The starting timestamp for filtering.
    - end_time: The ending timestamp for filtering.
    - key_timestamp: The column name for the timestamp, defaults to log_ids.start_time if not provided.

    Returns:
    - filtered_df: A DataFrame containing cases that start and end in the timespan.
    """

    # Set the key_timestamp to the provided value or use log_ids.start_time
    if key_timestamp is None:
        key_timestamp = log_ids.start_time

    # Use regular_cut to get the filtered DataFrame based on the start_time
    _, filtered_start_df = regular_cut(df, log_ids, start_time, cut_criterium='timestamp', key_timestamp=key_timestamp)

    # Use regular_cut to get the filtered DataFrame based on the end_time
    extracted_df, _ = intermediate_cut(filtered_start_df, log_ids, end_time, cut_criterium='timestamp', one_timestamp=False, key_timestamp=log_ids.end_time)

    return extracted_df


def extract_cut_to_time(df, log_ids, start_time, end_time, key_timestamp=None, one_timestamp=False):
    """
    Extracts rows from a DataFrame that fall within the specified start and end timestamps.

    Parameters:
    - df: The input DataFrame to filter.
    - log_ids: An object containing log identifiers, specifically the case identifier and timestamp column.
    - start_time: The starting timestamp for filtering.
    - end_time: The ending timestamp for filtering.
    - key_timestamp: The column name for the timestamp, defaults to log_ids.start_time if not provided.

    Returns:
    - filtered_df: A DataFrame containing cases that start and end in the timespan.
    """

    # Set the key_timestamp to the provided value or use log_ids.start_time
    if one_timestamp == False:
        key_timestamp_beginning = log_ids.start_time
        key_timestamp_ending    = log_ids.end_time
    else:
        key_timestamp_beginning = log_ids.end_time
        key_timestamp_ending    = log_ids.end_time

    # Use regular_cut to get the filtered DataFrame based on the start_time
    _, filtered_start_df = strict_cut(df, log_ids, start_time, cut_criterium='timestamp', key_timestamp=key_timestamp_beginning)

    # Use regular_cut to get the filtered DataFrame based on the end_time
    extracted_df, _ = strict_cut(filtered_start_df, log_ids, end_time, cut_criterium='timestamp', one_timestamp=False, key_timestamp=key_timestamp_ending)

    return extracted_df




def extract_split_logs(logs, log, log_name, target_dir_path, criterium_value, extract=False, cut_dates=None, write_out_logs = False):

    # Full log entry
    logs[log_name] = {}
    logs[log_name]['full_log'] = log

    if extract:
        start_time = cut_dates[0]
        end_time  = cut_dates[1]
        
        # Extract subsets of log data based on time intervals
        log_starting_in = extract_start_in_time(log, log_ids, start_time, end_time)
        log_contained_in = extract_contained_in_time(log, log_ids, start_time, end_time, one_timestamp=False)
        log_cut_to = extract_cut_to_time(log, log_ids, start_time, end_time, one_timestamp=False)

        # Organize the extractions in the dictionary
        logs[log_name]['extractions'] = {
            'starting': log_starting_in,
            'contained': log_contained_in,
            'cut': log_cut_to
        }
    else:
        logs[log_name]['extractions'] = {'full':log}

    # Initialize the splits dictionary
    logs[log_name]['splits'] = {}
    
    # Loop through each extraction type in the logs dictionary
    for extraction_type, extracted_df in logs[log_name]['extractions'].items():
        # Initialize a dictionary to hold split data for each extraction type
        logs[log_name]['splits'][extraction_type] = {}
        
        # Perform each type of split on the current extraction dataframe
        regular_train, regular_test           = regular_cut(extracted_df, log_ids, criterium_value, cut_criterium='nr_cases', one_timestamp=False, key_timestamp=log_ids.start_time)
        intermediate_train, intermediate_test = intermediate_cut(extracted_df, log_ids, criterium_value, cut_criterium='nr_cases', one_timestamp=False, key_timestamp=log_ids.start_time)
        strict_train, strict_test             = strict_cut(extracted_df, log_ids, criterium_value, cut_criterium='nr_cases', one_timestamp=False, key_timestamp=log_ids.start_time)
        
        # Store split results in the logs dictionary under the current extraction type
        logs[log_name]['splits'][extraction_type] = {
            'regular': {
                'train': regular_train,
                'test': regular_test
            },
            'intermediate': {
                'train': intermediate_train,
                'test': intermediate_test
            },
            'strict': {
                'train': strict_train,
                'test': strict_test
            }
        }

        # Directory and file writing section
        if write_out_logs:
            
            print('write_out_logs set to:', write_out_logs)
            # Writing each split type (regular, intermediate, strict) for the current extraction type to CSV
            for split_type, split_data in logs[log_name]['splits'][extraction_type].items():
                # Create directory path based on extraction and split type                            
                if extraction_type=='full':
                    split_path = os.path.join(target_dir_path, f"{log_name}_{split_type}")
                else:
                    split_path = os.path.join(target_dir_path, f"{log_name}_{extraction_type}_{split_type}")
                os.makedirs(split_path, exist_ok=True)
                print(f"Writing logs to: {split_path}")

                # Write train and test dataframes to CSV files
                # print(os.path.join(split_path, 'train_log.csv'))
                split_data['train'].to_csv(os.path.join(split_path, 'train_log.csv'), index=False)
                split_data['test'].to_csv(os.path.join(split_path, 'test_log.csv'), index=False)

    return logs




####### Evaluation functions

def align_column_names(df):
    if 'case:concept:name' in df.columns:
        df = df.rename(columns={'case:concept:name': 'case_id'})
    elif 'caseid' in df.columns:
        df = df.rename(columns={'caseid': 'case_id'})
    if 'Activity' in df.columns:
        df = df.rename(columns={'Activity': 'activity'})
    elif 'activity_name' in df.columns:
        df = df.rename(columns={'activity_name': 'activity'})
    elif 'task' in df.columns:
        df = df.rename(columns={'task': 'activity'})
    elif 'concept:name' in df.columns:
        df = df.rename(columns={'concept:name': 'activity'})
    if 'Resource' in df.columns:
        df = df.rename(columns={'Resource': 'resource'})
    elif 'user' in df.columns:
        df = df.rename(columns={'user': 'resource'})
    elif 'agent' in df.columns:
        if 'resource' in df.columns:
            df = df.drop(['resource'], axis=1)
        df = df.rename(columns={'agent': 'resource'})
    elif 'org:resource' in df.columns:
        df = df.rename(columns={'org:resource': 'resource'})
    if 'start_timestamp' in df.columns:
        df = df.rename(columns={'start_timestamp': 'start_time'})
    if 'end_timestamp' in df.columns:
        df = df.rename(columns={'end_timestamp': 'end_time'})
    # for SIMOD simulated logs
    if 'start_time' in df.columns:
        df = df.rename(columns={'start_time': 'start_time'})
    if 'end_time' in df.columns:
        df = df.rename(columns={'end_time': 'end_time'})
    if 'start:timestamp' in df.columns:
        df = df.rename(columns={'start:timestamp': 'start_time'})
    if 'time:timestamp' in df.columns:
        df = df.rename(columns={'time:timestamp': 'end_time'})
    return df


def highlight_min_max_from_means(results_df, mean_results):
    """
    Highlight cells in results_df based on min and max values from mean_results.
    Minimum mean values will be highlighted in green and maximum mean values in red.
    """
    def highlight(row):
        styles = []
        for col in results_df.columns:
            mean_value = mean_results.loc[row.name, col]
            cell_value = float(row[col].split(' ')[0])  # Extract the mean value from results_df
            
            if cell_value == mean_value:
                if cell_value == mean_results[col].min():
                    styles.append('background-color: green')
                elif cell_value == mean_results[col].max():
                    styles.append('background-color: red')
                else:
                    styles.append('')
            else:
                styles.append('')
        return styles
    
    return results_df.style.apply(highlight, axis=1)