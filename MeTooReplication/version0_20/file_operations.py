import os
from os.path import exists, isfile, join
import shutil
from pathlib import Path
import fileinput
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
import pickle
import time

def _combine_files(dest_directory, filepaths):
    ''' '''
    fn = '{}_comb.csv'.format(Path(filepaths[0]).stem)
    fp = os.path.join(dest_directory, fn)
    
    with open(fp, 'w') as file:
        input_lines = fileinput.input(filepaths)
        file.writelines(input_lines)
    file.close()
    
    for filepath in filepaths:
        os.remove(filepath)


'''
def _split_file(src_directory, dest_directory, filename, chunk_size, size_threshold):
    # delete
    times = []

    # Load data
    start = time.time()
    
    curr_filepath = "{}/{}".format(src_directory, filename)
    curr_file = open(curr_filepath, "rb")
    X_batch = pickle.load(curr_file)
    curr_file.close()

    end = time.time()
    times.append(("load data", end - start))

    # computations
    start = time.time() #

    X_batch_len = len(X_batch)
    num_rows, num_columns = X_batch.shape
    itemsize = X_batch.itemsize

    opt_rows_per_split = int(size_threshold / (num_columns * itemsize))
    chunks_per_split = int(opt_rows_per_split / chunk_size)

    end = time.time()
    times.append(("computations", end - start))

    # set up loop
    split_number = 1
    i = 0
    j = i + (chunk_size * chunks_per_split)
    while_cond = True

    while (while_cond):
        start = time.time()
        # approx_split_size = X_batch.iloc[i:j,:].memory_usage().sum()
        # approx_split_size = (j - i) * num_columns * itemsize

        # Write to a file if the split size is large enough
        # if (approx_split_size >= size_threshold) or (j >= X_batch_len):

        # Check if remainder of the data is undersized
        remaining =  X_batch_len - j

        if (remaining <= 0 and i != 0):
            return
        elif (remaining < chunk_size):
            curr_split  = X_batch[i:] # last batch
            while_cond = False
        else:
            curr_split = X_batch[i:j]
            i = j
            j = i + (chunk_size * chunks_per_split)

        # Write to a file
        curr_split_filepath = "{}/{}_split_{}.obj".format(
            dest_directory, 
            Path(filename).stem, 
            split_number
        )

        with open(curr_split_filepath, "wb") as curr_split_file:
            pickle.dump(curr_split, curr_split_file)

        # Future writes should be to a different file
        split_number += 1

        end = time.time()
        times.append(("iteration", end - start))
    
    print("length of {}: {} GB".format(filename, os.path.getsize(curr_filepath)/1000000000))
    print("times for {}: {}".format(filename, times))
'''


def _split_file(src_directory, dest_directory, filename, chunk_size, size_threshold):
    # chunk_size = # rows in IPCA batch
    # splits 2D arrays stored in a obj file

    # Open the data file to split into smaller files
    curr_filepath = os.path.join(src_directory, filename)
    curr_file = open(curr_filepath, "r")

    df = pd.read_csv(curr_filepath)

    i = 0
    j = chunk_size
    split_number = 1
    while_cond = True
    last_chunk = False

    while (while_cond):
        # Check if remainder is undersized
        if (len(df.index) - j) < chunk_size:
            j = len(df.index)
            last_chunk = True
            while_cond = False

        if (df.iloc[i:j,:].memory_usage().sum() >= size_threshold) or \
            last_chunk:
            # Current file split            
            curr_split_filepath = "{}_split_{}.csv".format(
                os.path.join(dest_directory, Path(filename).stem),
                split_number
            )

            temp_df = df.iloc[i:j,:]

            # write csv to the file
            temp_df.to_csv(curr_split_filepath, mode="w+", index=False)

            i = j
            j = i + chunk_size
            split_number += 1

        else:
            j += chunk_size


def split_files(src_directory, dest_directory, size_threshold=500000000, chunk_size=100, regenerate_all=True):
    ''' 
    Splits all files in a given directory into smaller files whose sizes are
    no more than given chunk size.

    Parameters:
    chunk_size (int): The largest file split size in bytes
    src_directory (string): The path to the directory containing files to split

    Returns:
    The directory path to the new file splits.
    '''
    if (not regenerate_all):
        return dest_directory

    # Get a list of files in the directory
    dir_contents = os.listdir(src_directory)
    files = [x for x in dir_contents if isfile(join(src_directory, x))]
        
    if (exists(dest_directory) == False):
        os.mkdir(dest_directory)
    elif (regenerate_all):
        # Clean up all previously split files
        shutil.rmtree(dest_directory)
        os.mkdir(dest_directory)

    # Split each file  
    for f in tqdm(files):
        _split_file(src_directory, dest_directory, f, chunk_size, size_threshold)
         
    # Return the new directory path
    return dest_directory


def get_files_in_dir(directory):
    '''
    Parameters
    ----------
    directory: Full path of directory to get files from.

    Returns
    -------
    array of filenames (excluding subdirectories) in directory.
    '''
    dir_contents = os.listdir(directory)
    files = [x for x in dir_contents if isfile(join(directory, x))]
    return files


def get_number_of_rows_in_csv(filepath):
    
    df = dd.read_csv(filepath)
    return len(df.index)


def print_filesizes(directory):
    '''
    Parameters
    ----------
    directory: Full path of directory.

    Returns
    -------
    array of filenames (excluding subdirectories) in directory.
    '''
    # Get a list of files in the directory
    dir_contents = os.listdir(directory)
    files = [x for x in dir_contents if isfile(join(directory, x))]
    
    for f in files:
        fp = join(directory, f)
        print("{}: {} bytes".format(f, os.path.getsize(fp)))


def print_num_rows_in_csvs(directory):
    for filename in get_files_in_dir(directory):
        num_rows = count_num_rows_in_csv(os.path.join(directory, filename))
        print("{}: {} rows".format(filename, num_rows))


def count_num_rows_in_csv(filepath):
    #curr_file = open(filepath, "r")
    df  = pd.read_csv(filepath)
    return len(df.index)