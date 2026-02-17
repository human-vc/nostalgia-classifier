# Data Sources & Pipeline

This directory contains scripts and documentation for building the complete
dataset needed for the nostalgia classifier and the downstream DiD analysis.

## Required Data Sources

### 1. Miller Center Presidential Speeches (Free)
- **URL:** https://millercenter.org/the-presidency/presidential-speeches
- **What:** Full transcripts of major presidential speeches, 1789–present
- **Use:** Stage 1 pre-fine-tuning corpus (domain adaptation on political language)
- **Script:** `scrape_miller_center.py` → `raw/miller_center_speeches.csv`

### 2. Wesleyan Media Project Political Ad Data ($20/year)
- **URL:** https://mediaproject.wesleyan.edu/dataaccess/
- **What:** Every broadcast TV political ad airing in 210 US media markets
- **Use:** Stage 2 training corpus (ad transcripts + metadata)
- **Access:** Academic only. Purchase 2020 and 2024 datasets.
- **Format:** Stata/SPSS files + .wmv/.mp4 video files
- **Key columns:** `ad_tone`, `creative`, `transcript`, `dma`, `party`, `race`
- **Note:** WMP data cannot be redistributed. Keep in `raw/` (gitignored).

### 3. MIT Election Data Lab — County Presidential Returns (Free)
- **URL:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
- **What:** County-level presidential election results, 2000–2020
- **Use:** Turnout calculation for DiD analysis
- **Script:** `fetch_election_data.py` → `processed/county_turnout.csv`

### 4. US Census ACS Demographics (Free, API key recommended)
- **URL:** https://api.census.gov/data/2022/acs/acs5
- **What:** County-level demographics (race, education, income)
- **Use:** Control variables in DiD regression
- **Script:** `fetch_demographics.py` → `processed/county_demographics.csv`
- **API Key:** Get one free at https://api.census.gov/data/key_signup.html

### 5. Nielsen DMA–County Crosswalk
- **Sources:**
  - UNC: https://dataverse.unc.edu (search "DMA county crosswalk")
  - USDA ERS: county adjacency + metro area files
  - Or build from Census CBSA definitions
- **Script:** `build_dma_mapping.py` → `processed/dma_county_map.csv`

## Pipeline Overview

```
1. scrape_miller_center.py     → raw/miller_center_speeches.csv
2. label_nostalgia.py          → processed/miller_center_labeled.csv  (Stage 1 data)
3. [manual] Purchase WMP data  → raw/wmp_2020.dta, raw/wmp_2024.dta
4. prepare_ad_corpus.py        → processed/ad_corpus_labeled.csv      (Stage 2 data)
5. fetch_election_data.py      → processed/county_turnout.csv
6. fetch_demographics.py       → processed/county_demographics.csv
7. build_dma_mapping.py        → processed/dma_county_map.csv
8. assemble_did_panel.py       → processed/did_panel.csv              (Final DiD dataset)
```

Run `../scripts/build_data.sh` to execute steps 1-2 and 5-7 automatically.
Steps 3-4 require WMP data (manual download after purchase).

## Directory Structure

```
data/
├── README.md                   ← You are here
├── scrape_miller_center.py     ← Speech transcript scraper
├── label_nostalgia.py          ← Rule-based nostalgia labeler
├── prepare_ad_corpus.py        ← WMP data processor
├── fetch_election_data.py      ← MIT Election Lab downloader
├── fetch_demographics.py       ← Census ACS API client
├── build_dma_mapping.py        ← DMA-county crosswalk builder
├── assemble_did_panel.py       ← Final panel dataset assembly
├── nostalgia_dictionary.json   ← Nostalgia lexicon (shared config)
├── raw/                        ← Raw downloaded data (gitignored)
└── processed/                  ← Cleaned, ready-to-use datasets
```

## Battleground States

The paper focuses on seven battleground states for the 2020 and 2024 cycles:
- Arizona, Georgia, Michigan, Nevada, North Carolina, Pennsylvania, Wisconsin

The pipeline filters to these states for the DiD analysis but keeps the full
national corpus for classifier training (more data = better model).
