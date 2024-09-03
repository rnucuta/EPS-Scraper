# from collections import defaultdict
from typing import Callable, List

from typing import List, Dict
# from fuzzywuzzy import fuzz
import pandas as pd
import re

from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)

from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement

from table_utils import ExtendedTableElement, EPSResult

class EPSTableExtractor(AbstractElementwiseProcessingStep):
    """
    Custom processing step to extract the latest EPS value from tables in the 10-Q filing.
    """
    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
    timeframe: int = 2020) -> None:
        super().__init__(
            types_to_process=types_to_process,
            types_to_exclude=types_to_exclude,
        )
        self.eps_results: List[EPSResult] = []
        self.timeframe  = timeframe

    def _process(
        self,
        elements: list[AbstractSemanticElement],
    ) -> list[AbstractSemanticElement]:
        self.eps_results = []
        for iteration in range(self._NUM_ITERATIONS):
            context = ElementProcessingContext(
                iteration=iteration,
            )
            self._process_recursively(elements, _context=context)

        return elements

    def _process_element(
        self,
        element: AbstractSemanticElement,
        _: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        Process method to extract EPS from tables and elements.
        """
        if isinstance(element, ExtendedTableElement):
            self._process_table(element)
        
        return element

    def _process_table(self, element: ExtendedTableElement):
        """
        Process tables to find EPS values using Pandas DataFrame.
        """
        # Convert the table to a Pandas DataFrame
        df = element.table_to_pandas()
        if df.empty:
            return

        # Clean up and preprocess the DataFrame for better parsing
        df = self._clean_dataframe(df)
        # pd.set_option('display.max_rows', None)  # Show all rows
        # pd.set_option('display.max_columns', None)
        # print(df) 

        eps_values = self._extract_eps_values(df)
        self.eps_results.extend(eps_values)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the DataFrame to standardize its format.
        """
        # Convert all elements to strings and strip whitespace
        df = df.fillna("").astype(str).map(str.strip)

        # Remove any fully empty rows or columns
        df = df.loc[:, (df != "").any(axis=0)]
        df = df[(df != "").any(axis=1)]
        return df

    def _extract_eps_values(self, df: pd.DataFrame) -> List[EPSResult]:
        """
        Extract all EPS values from the DataFrame based on robust rules and fuzzy matching.
        """
        eps_variants = ['eps', 'e.p.s', 'income per share', 'earnings per share', 'loss per share']

        eps_variants = [
            r'\b' + r'.*?'.join(re.escape(word) + r's?' for word in phrase.split()) + r'[\s:]*\b'
            for phrase in eps_variants
        ]
        
        found_eps = []

        header_row = None
        prev_index = -10
        reverse_order = False
        for row_idx, row in df.iterrows():
            leftmost_index = row[row != ''].first_valid_index()
            # Determine if this row is a header row for context
            if str(self.timeframe-1) in str(row) and str(self.timeframe) in str(row):
                if str(row).index(str(self.timeframe-1))<str(row).index(str(self.timeframe)):
                    reverse_order = True
                    # print("REVERSE ORDER")

            if not any(re.search(r'[+-]?\d*\.\d+', cell) for cell in row.astype(str)) and leftmost_index is not None:
                if prev_index == row_idx-1:
                    try:
                        header_row += " " + str(row.loc[leftmost_index]).strip()
                        prev_index=row_idx
                    except:
                        print(row)
                        print(leftmost_index)
                        print()
                else:
                    header_row = str(row.loc[leftmost_index]).strip()
                    prev_index = row_idx
                continue

            # Extract context information from the header row (if available)
            context = str(row.loc[leftmost_index]).strip()
            if header_row is not None:
                context = header_row + " " + context

            if not context:
                continue

            context = context.lower()

            df.columns = df.columns.to_flat_index()
            for col_idx, col in enumerate(reversed(df.columns) if reverse_order else df.columns):
                cell = str(row[col]).strip()
                old_context = str(context)
                if any(re.search(variant, context, re.IGNORECASE) for variant in eps_variants):
                    # set the parameters of the value being examined
                    eps_type = self._identify_eps_type(context)
                    if 'basic' in eps_type:
                        basic_type = 'basic'
                    elif 'diluted' in eps_type:
                        basic_type = 'diluted'
                    else:
                        basic_type = None
                    is_adjusted = 'non-gaap' in eps_type
                    is_loss = False
                    is_net = 'net' in eps_type

                    # Extract the EPS value
                    match = re.search(r'\s*[$]?\s*[\(\-]?\s*[$]?\s*\d+\.\d+\s*[$]?\s*[\)]?\s*[$]?\s*', str(cell))
                    if match:
                        value = float(re.sub(r'[(),$]', '', match.group())) * (-1 if '(' in match.group() else 1)
                        if value < 0:
                            is_loss = True
                        found_eps.append(EPSResult(value, basic_type, is_adjusted, is_loss, is_net, context))
                        # Stop searching in this row once match is found
                        break
                context = str(old_context)
        # Apply rules to determine the most relevant EPS value to prioritize
        return found_eps

    def _identify_eps_type(self, context: str) -> str:
        """
        Identify the type of EPS (basic, diluted, etc.) based on the context.
        """
        # Define regex patterns for context keywords
        keyword_checks = {
            'basic': 'basic',
            'diluted': 'diluted',
            ('net', 'total'): 'net',  # If either 'net' or 'total' is found
            # 'loss': 'loss',  # Specific phrase check
            # Regex pattern to match 'non-gaap', 'non gaap', 'adjusted'
            r'\b(non[- ]?g\.?a\.?a\.?p\.?|adjusted)\b': 'non-gaap',  
            # Regex pattern to match 'gaap', 'G.A.A.P.', 'unadjusted', etc., but not if it's part of 'non-gaap'
            r'\b(g\.?a\.?a\.?p\.?|unadjusted|un[- ]?adjusted|non[- ]?adjusted)\b': 'gaap',  
        }

        context_lower = context.lower()
        
        matched_results = []

        for check, result in keyword_checks.items():
            if isinstance(check, tuple): 
                if any(keyword in context_lower for keyword in check):
                    matched_results.append(result)
            else:
                if re.search(check, context_lower, re.IGNORECASE):
                    matched_results.append(result)
        
        # Remove 'gaap' if 'non-gaap' is present
        if 'non-gaap' in matched_results and 'gaap' in matched_results:
            matched_results.remove('gaap')
        
        return matched_results 

    def _add_eps_to_results(self, eps_value: EPSResult):
        """
        Add the identified EPS value to the results dictionary.
        """
        self.eps_results.extend(eps_value)

    def get_final_eps(self) -> EPSResult:
        """
        Determines the most specific EPS value using a prioritized sort.
        Returns the full EPSResult object.
        """
        if not self.eps_results:
            return 0 

        def weighted_sort_key(x):
            # Assign weights to each criteria
            score = 0
            if x.type=='basic':
                score+=100
            elif x.type=='diluted':
                score+=50
            # score += (100 if x.type == 'basic'  50)  # 'basic' gets a higher weight than 'diluted'
            score += (100 if not x.is_adjusted else 0)   # Unadjusted (GAAP) has a high priority
            score += (50 if x.is_net else 0)             # Net EPS has a moderate priority
            score += (50 if x.is_loss else 0)        # Prefer positive EPS
            # score += (x.value if not x.is_loss else -x.value)  # Use value as the last tie-breaker

            return score
        
        sorted_eps_results = sorted(self.eps_results, key=weighted_sort_key, reverse=True)

        # The first element in the sorted list is the most specific EPS
        return sorted_eps_results[0].value