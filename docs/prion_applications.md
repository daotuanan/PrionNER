# Applications of the Prion Disease NER + RE + KG System

This document summarizes practical applications of the current prion disease information extraction system, which combines named entity recognition, relation extraction design, and knowledge graph construction.

## 1. Literature Mining

- Extract structured biomedical information from prion disease papers, abstracts, and case reports.
- Identify mentions of:
  - prion disease subtypes,
  - symptoms,
  - diagnostic tests,
  - imaging and autopsy findings,
  - anatomic locations,
  - treatments,
  - complications,
  - epidemiologic measures.
- Convert unstructured clinical literature into searchable structured data.

## 2. Structured Case Summarization

- Convert case reports into structured records.
- Summarize each case by:
  - disease subtype,
  - presenting symptoms,
  - disease duration,
  - age at onset,
  - diagnostic tests,
  - abnormal findings,
  - treatment or supportive care,
  - complications,
  - iatrogenic exposure history if present.
- Support rapid review of rare and atypical presentations.

## 3. Prion Disease Knowledge Graph Construction

- Build a document-grounded knowledge graph for prion diseases.
- Represent relations such as:
  - disease -> symptom,
  - disease -> test,
  - disease -> finding,
  - finding -> anatomy,
  - test -> sensitivity/specificity,
  - disease -> incidence/prevalence,
  - disease -> exposure route.
- Support graph queries across the prion disease literature.

## 4. Clinical Phenotype Mapping

- Identify common and uncommon symptom patterns for:
  - `sCJD`,
  - `vCJD`,
  - `iCJD`,
  - `FFI`,
  - `GSS`,
  - `Kuru`.
- Compare subtype-specific clinical presentations.
- Help characterize atypical cases and phenotype variation.

## 5. Diagnostic Pathway Analysis

- Model the diagnostic workflow for suspected prion disease.
- Track links between:
  - disease and tests,
  - tests and findings,
  - findings and anatomy,
  - tests and reported performance measures.
- Support analysis of which tests are used most often and which findings are most strongly associated with each disease subtype.

## 6. Differential Diagnosis Support

- Extract diseases that are considered as alternatives to prion disease.
- Support clinical reasoning around rapidly progressive dementia and related syndromes.
- Help identify when prion disease is confused with:
  - stroke,
  - encephalitis,
  - autoimmune encephalitis,
  - Alzheimer's disease,
  - other neurologic or psychiatric conditions.

## 7. Epidemiology Summarization

- Capture incidence and prevalence values reported in the literature.
- Link epidemiologic measures to time points when available.
- Support surveillance-style summaries such as:
  - disease frequency by year,
  - reported burden in specific settings,
  - comparison of epidemiologic values across sources.

## 8. Test Performance Tracking

- Capture diagnostic performance claims for tests and modalities.
- Structure information such as:
  - sensitivity,
  - specificity,
  - test modality,
  - related disease context.
- Useful for imaging-focused and diagnostic criteria literature.

## 9. Iatrogenic Transmission and Exposure Analysis

- Identify medical procedures, grafts, or products associated with iatrogenic acquisition.
- Support extraction of evidence involving:
  - dura mater grafts,
  - corneal transplants,
  - growth hormone exposure,
  - biopsy-related or medical-product-related exposure routes.
- Useful for safety monitoring and historical transmission analysis.

## 10. Evidence Retrieval and Question Answering

- Support retrieval-augmented question answering over prion disease literature.
- Example question types:
  - Which symptoms are reported for `FFI`?
  - Which imaging findings are associated with `sCJD`?
  - Which anatomic sites are involved in `vCJD`?
  - Which tests show reported sensitivity or specificity for CJD?
- Answers can be grounded in source documents and extracted evidence.

## 11. Dataset Creation for Downstream Modeling

- Create labeled data for:
  - relation extraction,
  - evidence ranking,
  - document classification,
  - biomedical search,
  - graph-based reasoning.
- Support future model development beyond named entity recognition alone.

## 12. Research Trend and Evidence Synthesis

- Aggregate evidence across many publications.
- Identify recurring associations and underreported patterns.
- Support hypothesis generation, such as:
  - rare symptom clusters,
  - subtype-specific imaging profiles,
  - unusual anatomic involvement,
  - exposure-linked disease patterns.

## 13. Education and Clinical Reference Tools

- Build structured teaching resources for prion disease.
- Support clinician-facing reference tools summarizing:
  - hallmark symptoms,
  - typical and atypical findings,
  - diagnostic strategies,
  - major differential diagnoses,
  - epidemiologic context.

## 14. Current Limitations

- Fine-grained genetic reasoning is limited because genes and mutations are not fully modeled as dedicated entity types.
- Full test-result reasoning is limited because `Test_result` is not consistently present in the final released schema.
- Patient-level timeline reconstruction remains partial without richer event and case-level entities.
- Geographic and population-level epidemiology remains limited without dedicated location and population entities.

## 15. Best Near-Term Use Cases

The most realistic near-term applications with the current schema are:

- prion disease literature mining,
- structured case summarization,
- knowledge graph construction,
- differential diagnosis support,
- diagnostic pathway analysis,
- epidemiology summarization,
- test performance tracking,
- iatrogenic exposure analysis.
