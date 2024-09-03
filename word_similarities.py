from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

import nltk

import re

from table_utils import EPSResult


from sec_parser.semantic_elements.semantic_elements import (
    TextElement,
)
from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )

from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext
)

# Testing text
example_text = """Full Year Fiscal 2020 Adjusted EPS of $2.71, up 38% over prior year, with reported EPS of $1.84
Annual Revenues of $1.06 billion, up 14.5% over prior"""

class EPSTextExtractor(AbstractElementwiseProcessingStep):
    """
    Custom processing step to extract the latest EPS value from text in the 8-k filing.
    """

    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
    eps_results=None) -> None:
        super().__init__(
            types_to_process=types_to_process,
            types_to_exclude=types_to_exclude,
        )
        self.eps_results = eps_results
        self.SIMILARITY_THRESHOLD = 0.5
        self.tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
        self.model = AutoModel.from_pretrained('yiyanghkust/finbert-pretrain', output_attentions=True)
        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Similarity threshold for valid extraction
        self.SIMILARITY_THRESHOLD = 0.5

        self.eps_results: List[EPSResult]


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
        Process method to extract EPS from text elements. This would ideally be done
        for the first few lines of text since that is where the EPS is typically mentioned.
        Did not have time to make an abstraction around that for seq_parser
        to classify the beggining text sections of the document
        """

        # Process text elements (first few lines) to find potential EPS mentions
        if isinstance(element, TextElement):
            self._process_text(element)
        
        return element
    
    def _process_text(self, element: TextElement):
        """
        Process the first few lines of the filing for potential EPS data.
        """
        sentences = self.nltk_tokenizer.tokenize(element.text)
        for sentence in sentences:
            sentence = sentence.strip()
            # print(sentence)
            if not sentence:
                continue
            eps_variants = ['eps', 'e.p.s', 'income per share', 'earnings per share', 'loss per share']
            eps__regex_variants = [
            r'\b' + r'.*?'.join(re.escape(word) + r's?' for word in phrase.split()) + r'[\s:]*\b'
            for phrase in eps_variants
        ]
            
            # process sentence based on variant of the word EPS that is in the sentence
            for i, variant in enumerate(eps__regex_variants):
                if re.search(variant, sentence, re.IGNORECASE):
                    self.process_sentence(sentence, eps_variants[i])

    def process_sentence(self, element, eps_word):
        """
        Process a text element (sentence or paragraph) to extract and update the EPS value.
        """
        # Tokenize and encode the text element
        inputs = self.tokenizer(element, return_tensors="pt")

        # Extract token embeddings and attention weights from FinBERT
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            attentions = outputs.attentions  # Get attention weights

        token_embeddings = last_hidden_states.squeeze(0)  # Remove the batch dimension
        tokenized_text = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())  # Convert token IDs to strings

        # Identify tokens for relevant EPS variants
        eps_token_indices = [i for i, token in enumerate(tokenized_text) if any(variant in token.lower() for variant in eps_word.split())]

        dollar_prefixed_numbers = self._find_dollar_prefixed_numbers(tokenized_text)
        # print(dollar_prefixed_numbers)

        # Compute word embeddings for valid dollar numbers
        weighted_embeddings = []
        for number, idx in dollar_prefixed_numbers:
            number_embedding = token_embeddings[idx:idx + len(number.split())].mean(dim=0)  # Mean of token embeddings
            weight = attentions[-1].squeeze(0)[:, eps_token_indices, idx:idx + len(number.split())].mean()  # Average attention to number tokens
            weighted_embeddings.append((number_embedding * weight).numpy())

        # Convert to numpy array for similarity calculation
        if weighted_embeddings and not np.isnan(weighted_embeddings).any():  # Ensure there are valid dollar numbers found
            weighted_embeddings = np.array(weighted_embeddings)
            eps_embedding = token_embeddings[eps_token_indices].mean(dim=0).numpy()

            # Calculate cosine similarities between "EPS" embedding and weighted number embeddings
            similarities = cosine_similarity(eps_embedding.reshape(1, -1), weighted_embeddings)

            # Find the number most closely associated with "EPS"
            most_associated_number_index = similarities.argmax()
            most_similar_value = similarities[0, most_associated_number_index]

            # Check if the maximum similarity is above the threshold
            if most_similar_value > self.SIMILARITY_THRESHOLD:
                most_associated_number = dollar_prefixed_numbers[most_associated_number_index][0]
                
                eps_types = self._identify_eps_type(element)
                if 'basic' in eps_types:
                    basic_type = 'basic'
                elif 'diluted' in eps_types:
                    basic_type = 'diluted'
                else:
                    basic_type = None
                is_adjusted = 'non-gaap' in eps_types
                is_loss = False
                is_net = 'net' in eps_types
                if is_loss:
                    # Convert to a negative value if it belongs to 'loss per share' type
                    most_associated_number = '-' + most_associated_number if '-' not in most_associated_number else most_associated_number
                
                try:
                    self.eps_results.append(EPSResult(float(most_associated_number), basic_type, is_adjusted, is_loss, is_net, element))
                except:
                    print("Tried to update add ", most_associated_number)
    

    def _find_dollar_prefixed_numbers(self, tokenized_text):
        """
        Reconstruct and find all dollar-prefixed decimal numbers from tokenized text.
        """
        dollar_numbers = []
        i = 0
        while i < len(tokenized_text):
            if tokenized_text[i] == '$':
                number_parts = []
                i += 1 
                # Collect digits, decimals, and commas as part of the number
                while i < len(tokenized_text) and (tokenized_text[i].replace('.', '', 1).isdigit() or tokenized_text[i] in {'.', ','}):
                    number_parts.append(tokenized_text[i])
                    i += 1
                if number_parts:
                    reconstructed_number = ''.join(number_parts).replace(',', '')  

                    # # Ensure decimal number
                    if '.' in reconstructed_number:  
                        dollar_numbers.append((reconstructed_number, i - len(number_parts) - 1))  # Save the number and index
            else:
                i += 1
        return dollar_numbers
    
    def _identify_eps_type(self, context: str) -> str:
        """
        Identify the type of EPS (basic, diluted, etc.) based on the context.
        """
        # Define regex patterns for context keywords
        keyword_checks = {
            'basic': 'basic',
            'diluted': 'diluted',
            ('net', 'total'): 'net', 
            # 'loss': 'loss',  
            r'\b(non[- ]?g\.?a\.?a\.?p\.?|adjusted)\b': 'non-gaap',  
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
        
        if 'non-gaap' in matched_results and 'gaap' in matched_results:
            matched_results.remove('gaap')
        
        return matched_results 

    def get_final_eps(self) -> EPSResult:
        """
        Determines the most specific EPS value using a prioritized sort.
        Returns the full EPSResult object.
        """

        if not self.eps_results:
            return 0  # No EPS values found

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

        return sorted_eps_results[0].value