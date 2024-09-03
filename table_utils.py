import pandas as pd
from io import StringIO

from dataclasses import dataclass

from sec_parser.semantic_elements import (
    TextElement,
    TableElement,
    TitleElement,
    TopSectionTitle,
    NotYetClassifiedElement,
)

from sec_parser.utils.bs4_.table_to_markdown import TableToMarkdown

from sec_parser.utils.bs4_.get_single_table import get_single_table

from sec_parser.processing_engine.html_tag import HtmlTag
from sec_parser.processing_engine.processing_log import LogItemOrigin, ProcessingLog


from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)

from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement


@dataclass
class EPSResult:
    """Data structure to store EPS values with metadata."""
    value: float
    type: str
    is_adjusted: bool
    is_loss: bool
    is_net: bool
    context: str


# convert bs4 table to pandas dataframe for simplified parsing
class TableToPandas(TableToMarkdown):
    def convert_to_dataframe(self) -> str:
        tag = get_single_table(self._tag)
        unmerged = self._unmerge_cells(tag)
        pandas_table = pd.read_html(StringIO(str(unmerged)), flavor="lxml")[0]
        df = pandas_table.dropna(axis=1, how="all")
        return df.loc[:, df.apply(lambda col: col.dropna().astype(str).str.strip().str.len().gt(0).any())]

class ExtendedTableElement(TableElement):
    """
    Extends TableElement class to add pandas functionality to Table Element.
    """
    def __init__(
        self,
        html_tag: HtmlTag,
        *,
        processing_log: ProcessingLog | None = None,
        log_origin: LogItemOrigin | None = None,
    ) -> None:
        super().__init__(html_tag, processing_log=processing_log, log_origin=log_origin)
        self._pandas_table = None

    def table_to_pandas(self) -> pd.DataFrame:
        if not self._pandas_table:
            self._pandas_table = TableToPandas(self.html_tag._bs4).convert_to_dataframe()
        return self._pandas_table



class ExtendedTableClassifier(AbstractElementwiseProcessingStep):
    """
    TableClassifier class for converting elements into TableElement instances.

    This step scans through a list of semantic elements and changes it,
    primarily by replacing suitable candidates with TableElement instances.
    """

    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
    ) -> None:
        super().__init__(
            types_to_process=types_to_process,
            types_to_exclude=types_to_exclude,
        )
        self._row_count_threshold = 1

    def _process_element(
        self,
        element: AbstractSemanticElement,
        _: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        copied from TableClassifier, only setting table tags to extended table element for functionality
        """
        if element.html_tag.contains_tag("table", include_self=True):
            metrics = element.html_tag.get_approx_table_metrics()
            if metrics is None:
                element.processing_log.add_item(
                    log_origin=self.__class__.__name__,
                    message=("Skipping: Failed to get table metrics."),
                )
                return element
            if metrics.rows > self._row_count_threshold:
                return ExtendedTableElement.create_from_element(
                    element,
                    log_origin=self.__class__.__name__,
                )
            element.processing_log.add_item(
                log_origin=self.__class__.__name__,
                message=(
                    f"Skipping: Table has {metrics.rows} rows, which is below the "
                    f"threshold of {self._row_count_threshold}."
                ),
            )
        return element