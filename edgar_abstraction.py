#check if still needed at the end
from typing import Callable, List

from word_similarities import EPSTextExtractor
from table_utils import ExtendedTableElement, ExtendedTableClassifier
from table_similarities import EPSTableExtractor

# A lot of the imports below are unused. They are intentionally left here for reference 
# when using sec_parser library for if you were to uncomment out some of the parsing steps

from sec_parser.processing_engine.core import Edgar10QParser
from sec_parser.processing_steps.abstract_classes.abstract_processing_step import AbstractProcessingStep
from sec_parser.semantic_elements import (
    TextElement,
    TableElement,
    TitleElement,
    TopSectionTitle,
    NotYetClassifiedElement,
)
from sec_parser.processing_steps.empty_element_classifier import EmptyElementClassifier

from sec_parser.processing_steps.image_classifier import ImageClassifier

from sec_parser.processing_steps.individual_semantic_element_extractor.individual_semantic_element_extractor import (
    IndividualSemanticElementExtractor,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.image_check import (
    ImageCheck,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.table_check import (
    TableCheck,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.top_section_title_check import (
    TopSectionTitleCheck,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.xbrl_tag_check import (
    XbrlTagCheck,
)
from sec_parser.processing_steps.introductory_section_classifier import (
    IntroductorySectionElementClassifier,
)
from sec_parser.processing_steps.page_header_classifier import PageHeaderClassifier
from sec_parser.processing_steps.page_number_classifier import PageNumberClassifier
from sec_parser.processing_steps.supplementary_text_classifier import (
    SupplementaryTextClassifier,
)
from sec_parser.processing_steps.table_classifier import TableClassifier
from sec_parser.processing_steps.table_of_contents_classifier import (
    TableOfContentsClassifier,
)
from sec_parser.processing_steps.text_classifier import TextClassifier
from sec_parser.processing_steps.text_element_merger import TextElementMerger
from sec_parser.processing_steps.title_classifier import TitleClassifier
from sec_parser.processing_steps.top_section_manager_for_10q import (
    TopSectionManagerFor10Q,
)
from sec_parser.semantic_elements.composite_semantic_element import (
    CompositeSemanticElement,
)
from sec_parser.semantic_elements.highlighted_text_element import HighlightedTextElement
from sec_parser.semantic_elements.semantic_elements import (
    IrrelevantElement,
    NotYetClassifiedElement,
    TextElement,
)
from sec_parser.semantic_elements.table_element.table_element import TableElement

from sec_parser.semantic_elements.abstract_semantic_element import (
    AbstractSemanticElement,
)




class EnhancedEdgar8KParser(Edgar10QParser):
    """
        Extends abstraction provided by Edgar10Q parser for recursively parsing through html.
        Will go through each of the parsing steps. If use_embeddins is set to True, will use parse
        the text alongside the tables. All the information needed is in the table, so this flag is unnecessary.
        It was simply added as an area of interest.
    """
    def __init__(self, use_embeddings: bool, timeframe: int | None = False):
        super().__init__()
        # Initialize the EPSExtractor instance
        self.eps_extractor = EPSTableExtractor(types_to_process={ExtendedTableElement}, timeframe=timeframe)
        self.use_embeddings = use_embeddings
        if self.use_embeddings:
            self.text_extractor = EPSTextExtractor(types_to_process={TextElement}, eps_results = self.eps_extractor.eps_results)

    
    def parse(
        self,
        html: str | bytes,
        *,
        unwrap_elements: bool | None = None,
        include_containers: bool | None = None,
        include_irrelevant_elements: bool | None = None,
        ) -> list[AbstractSemanticElement]:
        """
            Recursively parse the html and return a list of semantic elements.
        """
        root_tags = self._html_tag_parser.parse(html)
        return self.parse_from_tags(
            root_tags,
            unwrap_elements=unwrap_elements,
            include_containers=include_containers,
            include_irrelevant_elements=include_irrelevant_elements,
        )
    
    def get_default_steps(
        self,
        get_checks: Callable[[], List[AbstractProcessingStep]] | None = None,
    ) -> List[AbstractProcessingStep]:
        """
            Returns a list of processing steps for the parse method. These steps are ran in order on every element.
            For example, the first step will run on every element, then the next step will run on every element, etc.
        """
        steps = [
            IndividualSemanticElementExtractor(
                get_checks=get_checks or self.get_default_single_element_checks,
            ),
            # ImageClassifier(types_to_process={NotYetClassifiedElement}),
            # EmptyElementClassifier(types_to_process={NotYetClassifiedElement}),
            ExtendedTableClassifier(types_to_process={NotYetClassifiedElement}),
            # TableOfContentsClassifier(types_to_process={ExtendedTableElement}),
            # TopSectionManagerFor10Q(types_to_process={NotYetClassifiedElement}),
            IntroductorySectionElementClassifier(),
            TextClassifier(types_to_process={NotYetClassifiedElement}),
            # HighlightedTextClassifier(types_to_process={TextElement}),
            # SupplementaryTextClassifier(
                # types_to_process={TextElement, HighlightedTextElement},
            # ),
            # PageHeaderClassifier(
            #     types_to_process={TextElement, HighlightedTextElement},
            # ),
            # PageNumberClassifier(
            #     types_to_process={TextElement, HighlightedTextElement},
            # ),
            TitleClassifier(types_to_process={HighlightedTextElement}),
            # TextElementMerger(),
            self.eps_extractor,
        ]
        if self.use_embeddings:
            steps.append(self.text_extractor)

        return steps

    def get_eps_values(self):
        """
        Returns the final EPS value extracted from all tables and text elements in the document.
        """
        # print(self.eps_extractor.eps_results)
        if len(self.eps_extractor.eps_results) == 0:
            if self.use_embeddings:
                return self.text_extractor.get_final_eps()
            else:
                return 0
        return self.eps_extractor.get_final_eps()
