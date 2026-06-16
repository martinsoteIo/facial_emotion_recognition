# Consolidated Facial Expression Dataset (Neuromorphic Computing Lab)

## Origin and Motivation
This dataset was curated following the professor's feedback to address critical limitations in the baseline models:
1. **Dataset 2 (Base):** Contained high class diversity but completely lacked the crucial `neutrality` baseline and the `fear` class.
2. **Dataset 1, 3 & 4:** Used as donor repositories to inject missing samples of `neutral` and `fear` expressions.

## Global Class Mapping (Ontology)
To prevent label collision and ensure consistency across repositories, all source annotations (`.txt` files) were mapped to the following global structure:
- `0`: angry (from Dataset 2)
- `1`: happy (from Dataset 2)
- `2`: sad (from Dataset 2)
- `3`: surprise (from Dataset 2)
- `4`: neutral (Injected from Dataset 1, 3, 4)
- `5`: fear (Injected from Dataset 1 and Dataset 4)

## Preprocessing Pipeline
- Original bounding boxes were preserved.
- File names were prefixed with `ds1_`, `ds3_`, and `ds4_` to avoid filename overwrites during the injection process.
