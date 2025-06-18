# IITI Document QA Pipeline

This project provides an end-to-end pipeline for scraping, preprocessing, and querying documents from the IITI website using Pathway and LLMs.

## Project Structure

- `data_preparation/` – Scripts for scraping and preprocessing data from the IITI website  
- `data/` – Directory for storing processed documents  
- `pathway_pipeline/` – Pathway-based pipeline for document retrieval and question answering  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

### 2.  Scrape and Preprocess Data

```bash
python data_preparation/scraper.py


### 3. Run the Pathway Pipeline

```bash
python pathway_pipeline/main.py

### 4. To test
curl -X POST http://localhost:8011      -H "Content-Type: application/json"      -d '{"messages": "<YOUR-MESSAGE>"}'