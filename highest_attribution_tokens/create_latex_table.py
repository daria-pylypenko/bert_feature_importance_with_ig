"""Turn text files into a latex table

Input:
    2 files with top tokens
    Number of rows to add to the table
 
      
Output:
    file with a latex table
"""

import argparse
import csv
import pandas as pd

from argparse import RawTextHelpFormatter

def prepare_frame(filename, title, rounding):
    """Prepare the dataframe

    Read it from file, rename colums,
    delete the index column,
    remove colons from the end of tokens,
    round the AAS values
    """
    frame = pd.read_csv(filename, header=None,
                    names=["index", "Token", "AAS"], delimiter=r"\s+", 
                    quoting=csv.QUOTE_NONE) 

    # Delete the index column
    frame = frame.drop("index", axis=1)

    # Delete the colon at the end of each word
    frame['Token'] = frame['Token'].map(lambda x: str(x)[:-1])

    # Round the AAS to 2 decimals
    frame['AAS'] = frame['AAS'].map(lambda x: round(x, rounding))

    # Create a multicolumn title
    frame.columns = pd.MultiIndex.from_product([[title], frame.columns])

    return frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_file_trans", help="Translationese file")
    parser.add_argument("input_file_org", help="Originals file")
    parser.add_argument("output_file", help="Joint latex table")
    parser.add_argument("-N", type=int, default=20, help="Number of rows in the table")
    parser.add_argument("-rounding", type=int, default=2, help="Number of decimals") # i.e. digits after the point
    args = parser.parse_args()

    frame_trans = prepare_frame(args.input_file_trans, "Translationese", args.rounding)
    frame_org = prepare_frame(args.input_file_org, "Original", args.rounding)

    # Join the frames
    frame = pd.concat([frame_trans, frame_org], axis=1)

    # Select first N rows
    frame = frame.head(n=args.N)

    # Rename index
    frame.index.name = "Rank"

    # Augment Rank to start from 1
    frame.index = frame.index + 1

    with open(args.output_file, 'w') as f:
        f.write(frame.to_latex())
