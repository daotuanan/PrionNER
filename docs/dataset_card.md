# Dataset Card

## Overview

PrionNER is a named entity recognition dataset for prion disease literature. The release package contains a canonical train/test split with both fine-grained and coarse-grained entity annotations, plus the raw BRAT source files and derived JSON and CoNLL exports.

## Splits

- Train: 247 documents
- Test: 70 documents
- Total: 317 documents

Each split is distributed in four parallel views:

- `data/raw/<split>/`: original paired `.txt` and `.ann` BRAT files
- `data/raw_text/<split>/`: trimmed text-only files
- `data/fine/`: fine-grained BRAT, JSON, and CoNLL exports
- `data/coarse/`: coarse-grained BRAT, JSON, and CoNLL exports

## Annotation Granularities

Fine-grained schema:

- 31 labels are defined in [prion_ner_entity_definitions_fine.json](../metadata/prion_ner_entity_definitions_fine.json)
- 30 labels are instantiated in the released data
- The schema-defined label `VPSPr` does not appear in the current train/test split

Coarse-grained schema:

- 15 labels are defined and instantiated in the released data
- See [prion_ner_entity_definitions_coarse.json](../metadata/prion_ner_entity_definitions_coarse.json)

## Entity Counts

Fine-grained:

- Train: 4,655 entities
- Test: 1,650 entities
- Discontinuous entities: 97 train, 34 test

Coarse-grained:

- Train: 4,655 entities
- Test: 1,650 entities
- Discontinuous entities: 97 train, 34 test

## Label Sets

Observed fine-grained labels:

`Age`, `Anatomic_location`, `Autopsy`, `Autopsy_finding`, `Blood_biomarker_test`, `Complication`, `Differential_Diagnosis`, `Duration`, `Electrophysio_test`, `FFI`, `GSS`, `Generic_Prion`, `Genetic_test`, `Imaging_finding`, `Imaging_sequence`, `Imaging_test`, `Incidence`, `Kuru`, `Molecular_assay`, `Prevalence`, `Sensitivity`, `Specificity`, `Symptom`, `Time_point`, `Treatment`, `fCJD`, `iCJD`, `sCJD`, `sFI`, `vCJD`

Observed coarse-grained labels:

`Acquired_Prion`, `Age`, `Anatomic_location`, `Complication`, `Differential_Diagnosis`, `Familial_Prion`, `Findings`, `Generic_Prion`, `Sequences`, `Sporadic_Prion`, `Stats`, `Symptom`, `Test_name`, `Time`, `Treatment`

## Special Annotation Structure

The dataset includes non-flat annotations, including discontinuous spans. Example figures are provided under [docs/figures](figures).
