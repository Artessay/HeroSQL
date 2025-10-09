import math
from typing import Optional

class OperatorIdMapper:
    """Provides mapping between operator name and unique operator id."""

    # Define operator types as class-level constant tuple
    OPERATOR_TYPES = (
        'Aggregate', 'AsofJoin', 'Calc', 'Correlate',
        'Exchange', 'Filter', 'Intersect', 'Join', 'Match',
        'Minus', 'Project', 'RepeatUnion', 'Snapshot',
        'Sort', 'SortExchange', 'TableFunctionScan',
        'TableModify', 'TableScan', 'TableSpool', 'Union',
        'Values', 'Window'
    )

    # Map from operator name to its unique ID
    OPERATOR_TO_ID_MAP = {
        operator: index
        for index, operator in enumerate(OPERATOR_TYPES)
    }

    NUM_OPERATORS = len(OPERATOR_TYPES)

    @classmethod
    def operator_to_id(cls, operator: str) -> Optional[int]:
        """
        Get the operator id for a given operator name.
        If not found, returns `default` (defaults to None) or raises a KeyError.
        """
        if operator in cls.OPERATOR_TO_ID_MAP:
            return cls.OPERATOR_TO_ID_MAP[operator]
        else:
            raise KeyError(f"Operator '{operator}' not found in OPERATOR_TYPES.")

    @classmethod
    def is_valid_operator(cls, operator: str) -> bool:
        """Check if the operator string is valid."""
        return operator in cls.OPERATOR_TO_ID_MAP
    

    
if __name__ == '__main__':
    
    # Example tests
    mapper = OperatorIdMapper

    # Valid lookup
    assert mapper.operator_to_id('Aggregate') == 0
    # print("Aggregate ->", mapper.operator_to_id('Aggregate'))
    print("Num operators:", mapper.NUM_OPERATORS)

    # Invalid lookup with exception
    try:
        mapper.operator_to_id('NonExistentOperator')
        assert False, "Expected KeyError for invalid operator lookup"
    except KeyError as e:
        # print("Expected exception:", e)
        pass

    # Check is_valid_operator
    assert mapper.is_valid_operator('Join')
    assert not mapper.is_valid_operator('FakeOp')

    print("All tests passed.")