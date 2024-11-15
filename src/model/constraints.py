import dis
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import sympy as sp
from pydantic import BaseModel

RTOL = 0.005
ATOL = 1

def is_close(a, b):
    return abs(a - b) <= RTOL * max(abs(a), abs(b)) + ATOL

class BaseConstraint(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, model: BaseModel) -> Optional[bool]:
        pass

class IsCloseConstraint(BaseConstraint):
    def __init__(self, name: str, func: Callable[[BaseModel], Tuple[float, float]]):
        super().__init__(name)
        self.func = func

    def evaluate(self, model: BaseModel) -> Optional[bool]:
        try:
            a, b = self.func(model)
            return is_close(float(a), float(b))
        except Exception:
            return None

class BoolConstraint(BaseConstraint):
    def __init__(self, name: str, func: Callable[[BaseModel], bool]):
        super().__init__(name)
        self.func = func

    def evaluate(self, model: BaseModel) -> Optional[bool]:
        try:
            return self.func(model)
        except Exception:
            return None

def get_involved_fields(constraint: Callable, model_instance: BaseModel) -> Set[str]:
    """Get the fields involved in a constraint.

    Args:
        constraint (Callable): A function that takes a model instance and returns a boolean.
        model_instance (BaseModel): The model instance to analyze.

    Returns:
        List[str]: The fields involved in the constraint.
    """
    code = constraint.__code__
    involved_fields = set()
    
    for instruction in dis.get_instructions(code):
        if instruction.opname == 'LOAD_ATTR':
            attr_name = instruction.argval
            if hasattr(model_instance, attr_name):
                involved_fields.add(attr_name)
    
    return set(involved_fields)

class ImplicitBaseModel(BaseModel):
    @classmethod
    def get_list_types(cls) -> Set[str]:
        return set()
    
    # Returns True if all constraints are valid or None, False if any constraint is invalid
    def is_valid(self) -> bool:
        details = self.get_constraint_sample_details()

        def recurse_dict(d: Dict[str, Union[Optional[bool], Dict, List[Union[bool, Dict]]]]) -> bool:
            for v in d.values():
                if v == False:
                    return False
                if isinstance(v, dict):
                    if recurse_dict(v) == False:
                        return False
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, bool):
                            if item == False:
                                return False
                        elif isinstance(item, dict):
                            if recurse_dict(item) == False:
                                return False
                        else:
                            raise ValueError(f'Unknown type: {type(item)}')
            return True
        
        return recurse_dict(details)
    
    def get_constraint_sample_details(self) -> Dict[str, Union[Optional[bool], Dict, List]]:
        constraints = self.get_constraints()
        out = {}
        for c in constraints:
            out[c.name] = c.evaluate(self)

        for list_field_name in self.get_list_types():
            list_field = getattr(self, list_field_name) or []
            list_field_results = []
            for i, item in enumerate(list_field):
                if isinstance(item, ImplicitBaseModel):
                    item_out = item.get_constraint_sample_details()
                    list_field_results.append(item_out)
            if len(list_field_results) > 0:
                out[list_field_name] = list_field_results

        return out
    
    def get_constraints(self) -> List[BaseConstraint]:
        raise NotImplementedError()

def solve_equation(equation_str: str, obj):
    # Convert the equation string to a SymPy equation
    eq = sp.Eq(*[sp.sympify(side) for side in equation_str.split('=')])

    # Get all symbols in the equation
    symbols_in_eq = eq.free_symbols
    total_symbols = len(symbols_in_eq)
    known_values = 0

    # Count known values and substitute them
    for symbol in symbols_in_eq:
        attr_name = symbol.name
        if hasattr(obj, attr_name) and getattr(obj, attr_name) is not None:
            known_values += 1
            attr_value = getattr(obj, attr_name)
            if isinstance(attr_value, list):
                eq = eq.subs(symbol, float(sum(attr_value)))
            else:
                eq = eq.subs(symbol, float(attr_value))

    # Check if we have enough known values (at least n-1)
    # NOTE: This is to avoid the edge cases where an equation a = b * c is resolved to 0 because b or c is 0 even though b or c is None
    if known_values < total_symbols - 1:
        return

    try:
        # Solve the equation with non-negative constraints
        solutions = sp.solve(eq, dict=True)

        # Filter out any solution that contains non-float values
        solutions = [s for s in solutions if all(isinstance(v, sp.Float) or isinstance(v, sp.Number) for v in s.values())]

        if len(solutions) != 1:
            return

        # Update object attributes with the solution
        for symbol, value in solutions[0].items():
            setattr(obj, symbol.name, Decimal(float(value)))
    except Exception as e:
        return
