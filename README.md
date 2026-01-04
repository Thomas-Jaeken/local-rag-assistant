# Local RAG Assistant

A local Retrieval-Augmented Generation (RAG) system workflow that is ready to load in your pdfs and generate answers grounded by those documents.
First, the documents need be processed and embedded once, using the ```all-MiniLM-L6-v2``` model. 
Then, the same model is used to embed the user query and the 5 chunks from the pdf data with the highest cosine similarity to the query are then used as context for an LLM together with the original query.
Since we want to run the LLM locally and without logins, I chose the```microsoft/phi-2``` model.

## Usage
Run only once:
```
python -m scripts.ingest  # include your pdfs in data/raw
python -m scripts.build_index
```
Run as many times as you want to prompt:
```
python -m scripts.query
```

## Example:
**query:** explain atmospheric transmission
**Answer:**
 Atmospheric transmission is the amount of light that passes through the atmosphere, which is affected by the Earth's atmosphere and the sun's radiation. The atmospheric transmission depends on the wavelength of the radiation, the atmospheric constituents, and the atmospheric conditions. The atmospheric transmission can be described by equations that involve the intensity of the incoming radiation, the spectral distribution of the incoming radiation, the absorption and scattering coefficients of the atmosphere, and the attenuation due to the Earth's atmosphere. The atmospheric transmission is important for the climate, the weather, the visibility, the refraction, and the absorption of radiation. Examples of the atmospheric transmission are the transmission of solar radiation by the atmosphere, the transmission of terrestrial radiation by the atmosphere, and the transmission of radiation by the clouds.