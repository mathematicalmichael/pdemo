import pandas as pd
import shutil


def apply() -> None:
    """Apply hard-coded peferences for using pandas"""
    pd.options.display.chop_threshold = None
    pd.options.display.colheader_justify = "right"
    pd.options.display.column_space = 12
    pd.options.display.date_dayfirst = False
    pd.options.display.date_yearfirst = False
    pd.options.display.encoding = "utf-8"
    pd.options.display.expand_frame_repr = True
    pd.options.display.float_format = None
    #
    pd.options.display.html.table_schema = False
    pd.options.display.html.border = 1
    pd.options.display.html.use_mathjax = True
    #
    pd.options.display.large_repr = "truncate"
    #
    pd.options.display.latex.repr = False
    pd.options.display.latex.escape = True
    pd.options.display.latex.longtable = False
    pd.options.display.latex.multicolumn = True
    pd.options.display.latex.multicolumn_format = "l"
    pd.options.display.latex.multirow = False
    #
    pd.options.display.max_categories = 20
    pd.options.display.max_columns = 250
    pd.options.display.max_colwidth = 50
    pd.options.display.max_info_columns = 100
    pd.options.display.max_info_rows = 1690785
    pd.options.display.max_rows = 100
    pd.options.display.max_seq_items = 100
    pd.options.display.min_rows = 10
    #
    pd.options.display.memory_usage = True
    pd.options.display.multi_sparse = True
    pd.options.display.notebook_repr_html = True
    pd.options.display.pprint_nest_depth = 3
    pd.options.display.precision = 4
    pd.options.display.show_dimensions = "truncate"
    #
    pd.options.display.unicode.east_asian_width = False
    pd.options.display.unicode.ambiguous_as_wide = False
    #
    pd.options.display.width = shutil.get_terminal_size((80, 20)).columns
