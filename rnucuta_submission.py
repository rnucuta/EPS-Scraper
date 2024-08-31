# Requirements:

# - needs to handle unseen filings
# - LATEST quarter Earnings Per Share, but check req num. 3


# 1. When both diluted EPS and basic EPS are present in the filing, prioritize outputting the basic EPS figure.

# 2. In cases where both adjusted EPS (Non-GAAP) and unadjusted EPS (GAAP) are provided
# in the filing, opt to output the unadjusted (GAAP) EPS value.

# 3. If the filing contains multiple instances of EPS data, output the net or total EPS.

# 4. Notably, enclosed figures in brackets indicate negative values. For instance, in the
# majority of filings, (4.5) signifies -4.5.

# 5. In scenarios where the filing lacks an earnings per share (EPS) value but features a loss
# per share, output values of the loss per share. Remember the output values should always
# be negative.

# output: filename, EPS

# Strategy: Can assume likely that edgar filings will have tables of similar format
# this gives generalizable strategy to attempt. Will also use regex search to "vote"
# on output


import os
import csv
from sec_parser.processing_engine import HtmlTagParser
from sec_parser.semantic_elements import TableElement
from sec_parser import Edgar10QParser
import bs4

# Directory containing the HTML files
directory = "./Training_Filings/"  # Adjust this path as needed

# Output CSV file
output_file = "output_eps.csv"





# Utility function, ignore it
def get_children_tags(source) -> list[bs4.Tag]:
    return [tag for tag in source.children if isinstance(tag, bs4.Tag)]


# Utility function, ignore it
def tag_to_string(tag):
    text = tag.text.strip()
    if len(text) > 0:
        text = text[:10] + "..." if len(text) > 10 else text
        return f"{tag.name} (text: {text})"
    else:
        return f"{tag.name} (no text)"


parse_result = bs4.BeautifulSoup(html, "lxml").html.body
bs4_tags = get_children_tags(parse_result)
for i, tag in enumerate(bs4_tags):
    print(f"Tag {i}: {tag_to_string(tag)}")


def extract_eps_from_table_element(table: TableElement):
    """
    Extract EPS from a given TableElement using known patterns.
    """
    # Assuming TableElement has a method to get all rows and cells
    eps_value = None
    
    # Iterate through rows and cells to find EPS data
    for row in table.get_rows():
        cells = row.get_cells()
        for cell in cells:
            # Check if the cell contains EPS-related text
            if "EPS" in cell.text or "earnings per share" in cell.text.lower():
                # Try to capture the value next to EPS-related text
                potential_values = [c.text for c in cells if c.text != cell.text]
                for value in potential_values:
                    if value:  # Ensure it's not empty
                        # Convert possible negative EPS values in brackets
                        if value.startswith('(') and value.endswith(')'):
                            value = f"-{value[1:-1]}"
                        # Clean value of commas and whitespace
                        value = value.replace(',', '').strip()
                        try:
                            eps_value = float(value)  # Try to parse the EPS value
                            return eps_value
                        except ValueError:
                            continue
    return eps_value

def extract_eps_from_html(file_path):
    """
    Extracts EPS data from an SEC HTML filing using sec-parser.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Parse the HTML content using sec-parser's HtmlTagParser
        parser = HtmlTagParser()
        parsed_html = parser.parse(html_content)

        # Identify all table elements in the parsed HTML
        tables = [element for element in parsed_html if isinstance(element, TableElement)]

        # Extract EPS from tables
        print(tables)
        for table in tables:
            eps = extract_eps_from_table_element(table)
            if eps is not None:
                return eps

        # If no EPS found in tables, return "N/A"
        return "N/A"

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return "Error"

def process_files(directory):
    """
    Processes all HTML files in the directory to extract EPS values and saves them to a CSV file.
    """
    # List to hold extracted data
    extracted_data = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            eps = extract_eps_from_html(file_path)
            extracted_data.append({"filename": filename, "EPS": eps})

    # Write results to CSV file
    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ["filename", "EPS"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in extracted_data:
            writer.writerow(data)
    
    print(f"EPS data has been written to {output_file}.")

# Run the file processing
process_files(directory)
