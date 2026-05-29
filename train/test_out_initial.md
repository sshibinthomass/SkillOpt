# Parameter Extraction Guidelines

## Objective
Extract the `mpn` field from a product or device description.

## Target Field
- `mpn`: The Manufacturer Part Number / manufacturer device identifier for the product.

## General Extraction Strategy
1. Read the full description and split it mentally on common delimiters such as:
   - `|`
   - `,`
   - spaces
   - `/` when used as a separator
2. Identify the token that looks like a formal manufacturer part number:
   - Usually an alphanumeric string
   - Often compact, with no descriptive words
   - May include symbols such as `*`, `-`, or other manufacturer-specific characters
3. Prefer the token that appears:
   - At the beginning of the record, or
   - As a standalone item between delimiters, especially near the manufacturer name
4. Ignore generic product descriptors such as:
   - `CAP`, `CER`, `CE`
   - capacitance, tolerance, voltage, dielectric, package size
   - manufacturer/brand names unless they are part of the actual part number

## How to Recognize the MPN
The `mpn` is typically the most specific identifier in the description. It often:
- Combines digits and letters in a structured pattern
- Appears exactly once or is repeated in multiple positions
- Is distinct from electrical specifications like `12pF`, `50V`, `X7R`, `0402`

### Common clues
In descriptions like:
- `04025A120F4T2A|CER 12pF 1% 50V C0G 0402|AVX KYOCERA|CAP`
- `CAP CE,1nF,10%,50V,X7R,0402,04025C102K4T*A,AVX|04025C102K4T*A|AVX|CAP`

The MPN is the structured token:
- `04025A120F4T2A`
- `04025C102K4T*A`

## Normalization and Precision Rules
Extract the MPN precisely as the manufacturer identifier, while removing non-essential packaging or branding additions.

### Keep
- The core manufacturer part number exactly as written
- Manufacturer-internal special characters if they are part of the true MPN, such as `*`

### Exclude
- Mounting codes or extra commercial ordering modifiers that are not part of the core MPN
- Tape-and-reel or packaging suffixes such as:
  - `-TR`
  - `/TR`
- Optional branding parameters or standalone brand/manufacturer tokens such as:
  - `AVX`
  - `KYOCERA`
- Generic product class labels such as:
  - `CAP`
  - `CER`
  - `CE`

## Disambiguation Rules
When multiple candidate tokens exist:
1. Prefer the standalone structured part number over descriptive text.
2. If the same part appears in multiple places, return the clean core version.
3. If one version includes packaging suffixes and another does not, return the version without packaging suffixes.
4. Do not substitute specifications or package size for the MPN.

## Common Mistakes to Avoid
- Do not return capacitance, voltage, dielectric, or package size as `mpn`.
- Do not include manufacturer names unless they are truly embedded in the part number.
- Do not append packaging suffixes like `-TR` or `/TR`.
- Do not strip valid internal characters that belong to the core MPN.

## Expected Output Format
Return a JSON object containing only the extracted target field when available:

- `{"mpn": "<extracted_mpn>"}`

If no valid MPN can be confidently identified, return:

- `{"mpn": null}`

## Example Outcomes
- `04025A120F4T2A|CER 12pF 1% 50V C0G 0402|AVX KYOCERA|CAP`
  - `{"mpn": "04025A120F4T2A"}`

- `CAP CE,1nF,10%,50V,X7R,0402,04025C102K4T*A,AVX|04025C102K4T*A|AVX|CAP`
  - `{"mpn": "04025C102K4T*A"}`

- `04025C222K4T2A|CER 2.2nF 10% 50V X7R 0402|AVX KYOCERA|CAP`
  - `{"mpn": "04025C222K4T2A"}`