# Parameter Extraction Guidelines

To extract parameters from the input description:
1. Identify the Manufacturer Part Number (MPN) or device identifier.
2. Extract the values for the following fields:
- mpn
- capacitance
- tolerance

## Rules

- Use only values explicitly present or reliably attributable to the identified part number/family.
- If any target value cannot be determined, return `null`.
- Do not hallucinate, estimate, or copy unrelated package/body text as electrical specs.
- Do not add extra commentary, confidence, or alternate values.
