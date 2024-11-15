import re
from decimal import Decimal
from typing import Any


# Special parsing for cnt field as it often has 3 trailing numbers which mess up the normal decimal parsing function
def parse_cnt_decimal(st):
    if isinstance(st, list) and len(st) > 0:
        return sum(parse_cnt_decimal(s) for s in st)
    
    if not st or st == 'None':
        return None
    
    if all(c == '-' for c in st):
        return Decimal(0)
    
    st = st.replace(',', '.')
    st = ''.join([c for c in st if c.isdigit() or c in ['.', '-']]).strip('.')

    if len(st.strip()) == 0:
        return None

    try:
        return Decimal(st)
    except:
        # print(f"Error converting {st} to Decimal")
        return None
    
def parse_decimal(st: str, is_percentage: bool = False):
    if isinstance(st, list) and len(st) > 0:
        return sum(parse_decimal(s) for s in st)
    
    if not st or st == 'None':
        return None
    
    if all(c == '-' for c in st):
        return Decimal(0)
    
    st = st.lower().replace('rp.', '')
    
    # Remove any whitespace and thousand separators (only when followed by exactly 3 digits)
    amount_str = re.sub(r'[\s,.](?=\d{3}(?!\d))', '', st.strip())

    contains_percentage = '%' in amount_str
    amount_str = ''.join([c for c in amount_str if c.isdigit() or c in ['.', ',', '-', '%']]).rstrip('.')
    
    # Add leading zero if string starts with decimal point
    if amount_str.startswith('.'):
        amount_str = '0' + amount_str
    
    # Check if the amount uses comma as decimal separator
    if ',' in amount_str and '.' not in amount_str:
        # Replace comma with period for proper float conversion
        amount_str = amount_str.replace(',', '.')
    elif ',' in amount_str and '.' in amount_str:
        # If both comma and period are present, assume comma is thousand separator
        amount_str = amount_str.replace(',', '')

    if amount_str.strip() == '':
        return None # st was 'ZERO RATED'

    # Move any trailing - to the front
    if amount_str[-1] == '-':
        amount_str = '-' + amount_str[:-1]
    
    # Check if there's a percentage sign and remove it
    amount_str = amount_str.replace('%', '')
    
    # Convert to Decimal
    try:
        value = Decimal(amount_str)
        # If it was a percentage, divide by 100
        if contains_percentage:
            value /= 100
        # If it was a percentage field and the value is >= 1, then convert it to the correct value
        if is_percentage and value >= 1:
            value /= 100
        return value
    except Exception as e:
        # print(f"Error converting {amount_str} to Decimal")
        return None

def parse_int(s):
    if isinstance(s, list):
        return sum(parse_int(s) for s in s)
    
    if not s or s == 'None':
        return None
    
    if all(c == '-' for c in s):
        return 0

    # Remove non number characters
    s = ''.join([c for c in s if c.isdigit() or c in ['.', ',', '-']])

    if s == '':
        return None

    # Move any trailing - to the front
    if s[-1] == '-':
        s = '-' + s[:-1]
    
    # Strip any trailing .
    # Some menu item had a '1.00.' value causing issues
    s = s.rstrip('.').rstrip(',')

    # CORD Train-141
    if s == '1,000':
        s = '1'

    try:
        return int(float(s))
    except:
        # print(f"Error converting {s} to int")
        return None

def parse_str(s):
    if isinstance(s, list):
        return parse_str(s[0])
    return s

def format_value(v: Any, value_if_none: str = '', decimals: int = 2) -> str:
    if isinstance(v, float) or isinstance(v, Decimal):
        try:
            # Convert to float and round to handle potential invalid Decimal states
            v = round(float(v), decimals)
            return f"{v:.{decimals}f}"
        except:
            return value_if_none
    elif isinstance(v, int):
        return str(v)
    elif isinstance(v, list):
        if len(v) == 0:
            return None
        return '|'.join([format_value(i) for i in v])
    elif v is None:
        return value_if_none
    return str(v)


if __name__ == '__main__':
    parse_decimal("RM.50")
    