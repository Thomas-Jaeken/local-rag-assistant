# Local RAG Assistant

A local Retrieval-Augmented Generation (RAG) system that performs
semantic search over a document corpus and generates grounded answers
using a local LLM on Apple Silicon (MPS).

## Features
- Document ingestion and chunking
- Sentence-transformer embeddings
- Vector similarity search
- Prompt-based answer generation
- Runs fully locally on macOS

## Usage

```
python -m scripts.ingest  # include your pdfs in data/raw
python -m scripts.build_index
python -m scripts.query
```

## Example:
*query:* explain atmospheric transmission
*Answer:*
 Atmospheric transmission is the amount of light that passes through the atmosphere, which is affected by the Earth's atmosphere and the sun's radiation. The atmospheric transmission depends on the wavelength of the radiation, the atmospheric constituents, and the atmospheric conditions. The atmospheric transmission can be described by equations that involve the intensity of the incoming radiation, the spectral distribution of the incoming radiation, the absorption and scattering coefficients of the atmosphere, and the attenuation due to the Earth's atmosphere. The atmospheric transmission is important for the climate, the weather, the visibility, the refraction, and the absorption of radiation. Examples of the atmospheric transmission are the transmission of solar radiation by the atmosphere, the transmission of terrestrial radiation by the atmosphere, and the transmission of radiation by the clouds.