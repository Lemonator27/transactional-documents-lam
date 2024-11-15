import unittest
from decimal import Decimal

from utils.utils import parse_decimal


class TestParseDecimal(unittest.TestCase):

    def test_basic_numbers(self):
        self.assertEqual(parse_decimal("123"), Decimal("123"))
        self.assertEqual(parse_decimal("123.45"), Decimal("123.45"))
        self.assertEqual(parse_decimal("-123.45"), Decimal("-123.45"))

    def test_thousand_separators(self):
        self.assertEqual(parse_decimal("1,234"), Decimal("1234"))
        self.assertEqual(parse_decimal("1,234,567"), Decimal("1234567"))
        self.assertEqual(parse_decimal("1.234.567"), Decimal("1234567"))

    def test_decimal_separators(self):
        self.assertEqual(parse_decimal("1,234.56"), Decimal("1234.56"))
        self.assertEqual(parse_decimal("1.234,56"), Decimal("1234.56"))

    def test_currency_symbols(self):
        self.assertEqual(parse_decimal("$123.45"), Decimal("123.45"))
        self.assertEqual(parse_decimal("€1,234.56"), Decimal("1234.56"))
        self.assertEqual(parse_decimal("1.234,56 €"), Decimal("1234.56"))

    def test_whitespace(self):
        self.assertEqual(parse_decimal(" 123.45 "), Decimal("123.45"))
        self.assertEqual(parse_decimal("\t1,234.56\n"), Decimal("1234.56"))

    def test_negative_numbers(self):
        self.assertEqual(parse_decimal("-123.45"), Decimal("-123.45"))
        self.assertEqual(parse_decimal("123.45-"), Decimal("-123.45"))

    def test_zero_values(self):
        self.assertEqual(parse_decimal("0"), Decimal("0"))
        self.assertEqual(parse_decimal("0.00"), Decimal("0"))
        self.assertEqual(parse_decimal("ZERO RATED"), None)

    def test_invalid_inputs(self):
        self.assertIsNone(parse_decimal(""))
        self.assertIsNone(parse_decimal("None"))
        self.assertIsNone(parse_decimal("abc"))

    def test_list_input(self):
        self.assertEqual(parse_decimal(["123.45"]), Decimal("123.45"))
        self.assertIsNone(parse_decimal([]))

    def test_edge_cases(self):
        self.assertEqual(parse_decimal("1,234,567.89"), Decimal("1234567.89"))
        self.assertEqual(parse_decimal("1.234.567,89"), Decimal("1234567.89"))
        self.assertEqual(parse_decimal("1,200"), Decimal("1200"))
        self.assertEqual(parse_decimal("4.1000"), Decimal("4.1"))
        self.assertEqual(parse_decimal(".5"), Decimal("0.5")) 
        self.assertEqual(parse_decimal(".21"), Decimal("0.21")) 
        self.assertEqual(parse_decimal("RM.50"), Decimal("0.50")) 
        self.assertEqual(parse_decimal("-.5"), Decimal("-0.5"))
        self.assertEqual(parse_decimal("Rp.500.000"), Decimal("500000"))