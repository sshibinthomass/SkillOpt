import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skillopt.envs.generic_csv.adapter import parse_engineering_value

# Test cases
test_cases = [
    ("1:00 AM", (1.0, "a")),
    ("3:00 AM", (3.0, "a")),
    ("12:00 AM", (12.0, "a")),
    ("12:00 PM", (12.0, "a")),
    ("1 A", (1.0, "a")),
    ("1000 mA", (1.0, "a")),
    ("3A", (3.0, "a")),
]

for tc, expected in test_cases:
    res = parse_engineering_value(tc)
    assert res == expected, f"Failed for {tc}: got {res}, expected {expected}"
print("All adapter parsing tests passed successfully!")
