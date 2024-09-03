import argparse
import os
from edgar_abstraction import EnhancedEdgar8KParser
import csv
from tqdm import tqdm
import nltk

#usage: python3 rnucuta_submission.py --input_dir "Training_Filings" --output_file "output_eps.csv"
#to try out the word embeddings feature, use the --use_embeddings flag
#this has an accuracy of 22% on the test set, not intended for actual use


def validate_directory(path):
    """Check if the path is a valid directory."""
    if not os.path.isdir(path):
        absolute_path = os.path.abspath(path)

        parent_dir = os.path.dirname(absolute_path)
        
        if parent_dir and not os.path.isdir(parent_dir):
            raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory path.")
    return path

def validate_file_path(path):
    """Check if the path is a valid file path format."""
    if os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is a directory, not a file path.")

    absolute_path = os.path.abspath(path)

    parent_dir = os.path.dirname(absolute_path)
    
    if parent_dir and not os.path.isdir(parent_dir):
        raise argparse.ArgumentTypeError(f"The directory '{parent_dir}' does not exist.")
    
    return path

def parse_args():
    """Options for the parser."""
    parser = argparse.ArgumentParser(description="Parse input and output file paths with an optional embeddings flag.")
    parser.add_argument('--input_dir', default='Training_Filings', type=validate_directory, help="The input file directory path.")
    parser.add_argument('--output_file', default='output_eps.csv', type=validate_file_path, help="The output file directory path.")
    parser.add_argument('--use_embeddings', default=False, action='store_true',help="Flag to indicate if the text should be processed (Not just the tables). If set, it is true.")
    parser.add_argument('--timeframe', default=2020, type=int, help="The desired year of the quarterly EPS.")
    args = parser.parse_args()
    return args


def process_filings(input_folder: str, output_csv: str, use_embeddings: bool, timeframe: int):
    """Initialize the parser and run through every file one by one."""
    output_data = []

    # Iterate through the HTML files in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.html'):
            file_path = os.path.join(input_folder, filename)
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = EnhancedEdgar8KParser(use_embeddings, timeframe)

            # Parse the HTML content using Edgar10QParser
            parser.parse(html_content)

            # Retrieve EPS values
            eps_value = parser.get_eps_values()

            output_data.append((filename, eps_value))

    # Write the output to a CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'EPS'])
        csv_writer.writerows(output_data)
    
    print("Done! Check {} for the output!".format(output_csv))

if __name__ == "__main__":
    args = parse_args()
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    process_filings(args.input_dir, args.output_file, args.use_embeddings, args.timeframe)