# Prion Relation Schema

This document lists the proposed relation types for converting PrionNER into a relation extraction dataset and clinical knowledge graph focused on prion disease findings.

| Relation Name | Head | Tail | Definition |
| --- | --- | --- | --- |
| `has_symptom` | `Prion_Disease` | `Symptom` | Links a prion disease mention to a symptom or clinical manifestation explicitly described as part of the disease presentation. |
| `has_complication` | `Prion_Disease` | `Complication` | Links a prion disease mention to a complication, adverse outcome, or consequence that occurs during the disease course. |
| `has_differential_diagnosis` | `Prion_Disease` | `Differential_Diagnosis` | Links a prion disease mention to a non-prion disorder discussed as an alternative or competing diagnosis. |
| `has_incidence` | `Prion_Disease` | `Incidence` | Links a prion disease mention to an incidence value describing the rate of new cases in a population or setting. |
| `has_prevalence` | `Prion_Disease` | `Prevalence` | Links a prion disease mention to a prevalence value describing how common the disease is in a population or setting. |
| `evaluated_by` | `Prion_Disease` | `Test` | Links a prion disease mention to a diagnostic test, assay, examination, or pathology procedure used to assess or diagnose it. |
| `targets_anatomic_location` | `Test` | `Anatomic_location` | Links a diagnostic test or procedure to the anatomical location, tissue, organ, or body compartment it examines or samples, such as `MRI -> brain` in "brain MRI". |
| `has_sensitivity` | `Test` | `Sensitivity` | Links a diagnostic test or procedure to a sensitivity value explicitly reported for that test. |
| `has_specificity` | `Test` | `Specificity` | Links a diagnostic test or procedure to a specificity value explicitly reported for that test. |
| `has_finding` | `Prion_Disease` | `Finding` | Links a prion disease mention to an imaging or autopsy finding explicitly associated with that disease. |
| `involves_anatomic_location` | `Prion_Disease` | `Anatomic_location` | Links a prion disease mention directly to an anatomical system, tissue, or structure described as being involved or affected, even when no separate finding span is annotated. |
| `found_by` | `Finding` | `Test` | Links a finding to the test or procedure by which that finding was observed or established. |
| `located_in` | `Finding_or_Symptom` | `Anatomic_location` | Links a symptom, imaging finding, or autopsy finding to the anatomical location where it is observed, localized, or most strongly associated. |
| `observed_on_sequence` | `Imaging_finding` | `Imaging_sequence` | Links an imaging finding to the imaging sequence on which the finding is reported or best visualized. |
| `has_age_of_onset` | `Prion_Disease` | `Age` | Links a prion disease mention to an age expression describing patient age or age at onset. |
| `has_duration` | `Prion_Disease` | `Duration` | Links a prion disease mention to a duration expression describing disease course, illness duration, or progression interval. |
| `measured_at_time` | `Incidence_or_Prevalence` | `Time_point` | Links an epidemiologic measure to the specific year or time point at which it was measured or reported. |
| `occurs_at_time` | `Clinical_Event` | `Time_point_or_Duration` | Links a symptom, test, treatment, or complication to a time point or duration when that event occurs in the clinical course. |
| `treated_with` | `Prion_Disease_or_Symptom` | `Treatment` | Links a prion disease or symptom mention to a treatment, supportive intervention, or medication used for management. |
| `acquired_via` | `Prion_Disease` | `Treatment` | Links a prion disease mention to an iatrogenic exposure source, medical product, graft, or procedure through which the disease was acquired when that exposure is annotated under the existing `Treatment` label. |
| `is_synonym` | `Abbreviation or same surface form` | `Same-type full text or same surface form` | Links one mention to another same-type synonymous mention. For different strings, use abbreviation to full text, such as `CJD -> Creutzfeldt-Jakob disease`; identical surface-form pairs like `CJD -> CJD` are also allowed across different mentions. |

## Notes

- These relations are intended for explicit mention-level annotation, not inferred background knowledge.
- Relation direction should be kept consistent during annotation and model training.
- In all exported triples, the `Head` column should be treated as the source node and the `Tail` column as the target node.
- The `acquired_via` relation exists because some mentions currently labeled as `Treatment` in the corpus are actually exposure routes or iatrogenic sources rather than therapies.
- Ontology-level subtype relations such as `FFI subtype_of Familial_Prion` or `vCJD subtype_of Acquired_Prion` should be handled separately from mention-level evidence relations.
- Mention-level synonym evidence such as `CJD is_synonym Creutzfeldt-Jakob disease` or `CJD is_synonym CJD` should remain distinct from ontology-level disease hierarchy edges.

## Knowledge Graph Design Notes

The points below should be recorded before building the relation extraction dataset and the downstream prion disease knowledge graph. These decisions will directly affect graph consistency, node merging, edge meaning, and scientific usability.

### 1. Separate ontology facts from evidence facts

- Ontology facts are schema-level truths that do not depend on a specific document.
- Evidence facts are extracted from a document, sentence, or case report.
- Example ontology fact: `vCJD subtype_of Acquired_Prion`.
- Example evidence fact: `vCJD has_finding pulvinar sign`.
- These two layers should not be stored as if they were the same type of knowledge.
- Recommended practice:
  - Keep a canonical ontology layer for disease hierarchy and stable concept relations.
  - Keep an evidence layer for document-grounded triples with provenance.

### 2. Define the unit of the graph clearly

- Decide whether nodes represent:
  - canonical biomedical concepts,
  - document mentions,
  - patient-level facts,
  - or paper-level summary statements.
- A practical design is to maintain two linked layers:
  - `Concept layer`: normalized nodes such as `sCJD`, `MRI`, `pulvinar sign`, `thalamus`.
  - `Evidence layer`: mention-level or sentence-level assertions linked back to source text.
- Without this separation, review-level claims and single-patient observations will be mixed together and may be misleading.

### 3. Normalize entity names carefully

- Synonyms and abbreviations must be mapped consistently.
- Example:
  - `Creutzfeldt-Jakob disease`
  - `CJD`
  - `sporadic CJD`
  - `sCJD`
- These should not all collapse to one node automatically.
- Recommended rule:
  - Merge exact synonyms only when they refer to the same clinical concept.
  - Keep subtype nodes separate from parent disease nodes.
  - Preserve original surface forms in the evidence layer even after normalization.

### 4. Preserve disease granularity

- The schema contains both broad and specific disease labels.
- Important distinctions include:
  - `Generic_Prion`
  - `sCJD`
  - `vCJD`
  - `iCJD`
  - `FFI`
  - `GSS`
  - `Kuru`
- KG construction must preserve this hierarchy.
- Do not merge all subtype mentions into a generic `CJD` node.
- Recommended practice:
  - Create canonical subtype nodes.
  - Link them to broader parents through ontology edges such as `subtype_of`.

### 5. Fix head-tail direction once

- The relation table defines one direction for each edge.
- That direction should remain identical across:
  - annotation,
  - model training,
  - export scripts,
  - KG loading,
  - downstream querying.
- Example:
  - use `Prion_Disease -> Symptom` for `has_symptom`
  - not both `Disease -> Symptom` and `Symptom -> Disease`
- If inverse queries are needed, generate them at query time rather than mixing directions in the annotated data.

### 6. Record provenance for every evidence triple

- Every extracted relation should retain:
  - document ID or filename,
  - source split,
  - sentence or text span,
  - head span offsets,
  - tail span offsets,
  - relation label,
  - annotation confidence if available.
- Recommended minimum provenance fields:
  - `doc_id`
  - `source_text`
  - `head_text`
  - `tail_text`
  - `head_offsets`
  - `tail_offsets`
  - `relation`
- Provenance is essential for:
  - auditability,
  - curator review,
  - deduplication,
  - confidence scoring,
  - and scientific trust.

### 7. Model negation, uncertainty, and diagnostic status

- Clinical texts do not only contain positive findings.
- They also contain:
  - suspected diagnoses,
  - ruled-out conditions,
  - differential diagnoses,
  - negative test results,
  - uncertain interpretations.
- Example patterns:
  - "CJD was suspected"
  - "stroke was considered"
  - "panel was negative"
  - "findings may represent"
- Recommended practice:
  - Keep `Differential_Diagnosis` relations distinct from asserted disease relations.
  - Add assertion metadata later if possible, such as:
    - `asserted`
    - `suspected`
    - `negated`
    - `differential`
    - `historical`
- If assertion metadata is not captured, the KG may overstate claims.

### 8. Capture temporality explicitly

- Many facts in prion disease literature are time-bound.
- Time affects:
  - onset,
  - progression,
  - duration,
  - order of symptoms,
  - test timing,
  - epidemiologic measures,
  - and exposure history.
- The current schema already supports:
  - `Age`
  - `Duration`
  - `Time_point`
- Recommended practice:
  - keep time-linked edges such as:
    - `has_age_of_onset`
    - `has_duration`
    - `occurs_at_time`
    - `measured_at_time`
  - do not discard these during export.
- If time is dropped, the resulting KG will lose much of the clinical course information.

### 9. Distinguish review articles from patient-level case evidence

- Some documents summarize the literature or diagnostic criteria.
- Others describe one patient or a small case series.
- These are not equivalent evidence sources.
- Example difference:
  - a review may state general diagnostic performance,
  - a case report may describe one atypical symptom sequence.
- Recommended practice:
  - store document type metadata when possible:
    - `case_report`
    - `case_series`
    - `review`
    - `epidemiology`
    - `methods`
- This allows filtering the graph by evidence type later.

### 10. Handle overloaded entity labels carefully

- Some current entity labels are semantically broader than their names suggest.
- The most important example is `Treatment`.
- In this corpus, `Treatment` can represent:
  - genuine therapies,
  - supportive care,
  - medications,
  - iatrogenic exposure sources,
  - grafts,
  - medical products,
  - procedures associated with transmission.
- Because of this, the KG should not assume every `Treatment` node is therapeutic.
- Recommended practice:
  - retain `treated_with` for true management relations,
  - retain `acquired_via` for iatrogenic transmission or exposure relations,
  - consider future schema refinement into:
    - `Therapy`
    - `Supportive_Care`
    - `Exposure_Source`
    - `Medical_Procedure`

### 11. Account for schema drift between config files and released annotations

- The released `.ann` files and metadata are more reliable than the older root config.
- Some legacy labels in older config files do not match the current final annotation set.
- Recommended practice:
  - treat the released final `.ann` files plus final metadata as the source of truth,
  - do not design the KG solely from the older root `annotation.conf`.

### 12. Support discontinuous spans and non-flat annotations

- The dataset includes discontinuous entities.
- Some clinical mentions are split across text spans.
- Recommended practice:
  - keep the original span fragments in evidence storage,
  - generate one normalized mention string for KG mapping,
  - avoid dropping discontinuous entities during preprocessing.
- If discontinuous spans are flattened incorrectly, relation attachment may become wrong.

### 13. Direct disease-to-anatomy relations are sometimes necessary

- Not all anatomy mentions are mediated by a separate finding entity.
- Some texts directly state that a disease involves a system or organ.
- Example:
  - a disease involving the central nervous system,
  - vCJD involving the lymphoreticular system or tonsils.
- This is why `involves_anatomic_location` is useful in addition to:
  - `has_finding`
  - `located_in`
  - `targets_anatomic_location`

### 14. Epidemiology and test-performance facts need their own treatment

- Epidemiology values are not the same as patient-level symptoms.
- Test performance values are not the same as findings.
- Recommended separate edge types:
  - `has_incidence`
  - `has_prevalence`
  - `has_sensitivity`
  - `has_specificity`
  - `measured_at_time`
- These facts are especially important for:
  - surveillance summaries,
  - diagnostic guideline papers,
  - and benchmark test literature.

### 15. Missing entity types will limit future KG expressiveness

- Some high-value biomedical relations cannot be represented well with the current entity set.
- Important missing entity families include:
  - `Gene`
  - `Mutation`
  - `Protein`
  - `Specimen`
  - `Exposure`
  - `Procedure`
  - `Geography`
  - `Population`
  - `Family_History`
  - `Patient_Case`
- Without these, some relations will remain approximate or overloaded.
- Example limitations:
  - gene-mutation-disease associations,
  - specimen-based assay interpretation,
  - country-level epidemiology,
  - family-history risk factors,
  - transmission pathway modeling.

### 16. Plan for confidence scoring and curator review

- Automatically extracted relations should not be treated as equally reliable.
- Recommended practice:
  - assign confidence scores from the RE model,
  - keep source evidence visible to human reviewers,
  - allow curator validation for high-value relations.
- This is particularly important for:
  - rare subtype claims,
  - atypical symptom relations,
  - exposure pathways,
  - and differential diagnosis edges.

### 17. Design for deduplication without losing evidence

- The same fact may appear:
  - multiple times in one document,
  - across train and test-style sources,
  - across review and case papers,
  - under slightly different wording.
- Recommended practice:
  - deduplicate at the concept-triple layer,
  - but keep all supporting evidence records linked to that canonical edge.
- This produces a KG that is both compact and evidence-rich.

### 18. Recommended KG output structure

- A robust output design is:
  - `nodes.tsv` for canonical normalized nodes,
  - `edges.tsv` for normalized concept-level edges,
  - `evidence.tsv` for mention-level supporting evidence,
  - `ontology.tsv` for schema or hierarchy relations.
- Suggested node fields:
  - `node_id`
  - `node_type`
  - `canonical_name`
  - `synonyms`
- Suggested edge fields:
  - `edge_id`
  - `head_id`
  - `relation`
  - `tail_id`
  - `evidence_count`
- Suggested evidence fields:
  - `evidence_id`
  - `edge_id`
  - `doc_id`
  - `sentence_text`
  - `head_span`
  - `tail_span`
  - `assertion_status`
  - `confidence`

### 19. Minimal recommendations before KG construction

- Before building the first KG version, define:
  - the canonical node inventory,
  - the normalization policy for disease aliases,
  - the assertion/uncertainty handling policy,
  - the evidence provenance format,
  - and the rule for handling overloaded `Treatment` mentions.
- If these are not fixed first, the graph will be harder to clean later than to design correctly now.
