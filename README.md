# Multi-Temporal Cropland Classification from Sentinel-2 L2A Imagery

A comparison of classical ML and deep learning approaches to cropland
classification across three agricultural regions of India, using
multi-temporal Sentinel-2 imagery pulled via Google Earth Engine.

## Problem statement

Given multi-season Sentinel-2 L2A composites over agricultural land, classify
image patches as **cropland vs. non-cropland**, and compare three modeling
approaches, a classical ML baseline, a from-scratch deep segmentation
model, and a pretrained geospatial foundation model on the same held-out
test set.

## Scope limitation (read first)

This project performs **binary cropland/non-cropland classification, not
crop-species classification** (e.g. wheat vs. rice vs. cotton). During
Stage 2, I checked three candidate label sources for India-specific
crop-type ground truth and found none that could be scripted against these
AOIs:

- **Bhuvan LULC** (ISRO/NRSC): only available via manual portal download,
  no API; land-cover categories are broad (e.g. "agricultural"), not
  crop-species; most recent full-country vintage is 2015-16.
- **ICRISAT**: publishes **district-level tabular** crop area/production
  statistics — not per-pixel labels, so it can't be rasterized to align
  with 64×64 image patches.
- No verified Kaggle/Zindi dataset was found with confirmed pixel overlap
  to these three specific AOIs.

**Fallback used:** [ESA WorldCover v200](https://esa-worldcover.org/) (10m,
2021), class 40 ("Cropland"), pulled directly in GEE — no separate
registration needed. This gives a defensible, scriptable, pixel-aligned
binary label but caps the project's ceiling at presence/absence of
cropland, not crop identity.


## Data sources

| Source | Role | Access method |
|---|---|---|
| Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`) | Input imagery, 4 seasonal composites × 3 regions | Google Earth Engine Python API |
| `COPERNICUS/S2_CLOUD_PROBABILITY` | Cloud/shadow masking (s2cloudless) | Google Earth Engine |
| ESA WorldCover v200 | Cropland ground-truth mask | Google Earth Engine |

Regions: representative ~150-350 km² agricultural sub-areas in Punjab
(Ludhiana belt), Maharashtra (Nashik belt), and Karnataka (Raichur belt) —
not full states, to keep tile count and export size manageable.

## Methodology

1. **Acquisition** (Stage 1): Sentinel-2 L2A pulled for 4 seasonal windows
   per region spanning the Kharif/Rabi crop calendar, cloud-masked with the
   s2cloudless probability method (QA60 alone was avoided, it's known to
   be unreliable/empty for scenes processed under ESA's post-2022
   baseline), median-composited per season, exported as 10-band GeoTIFFs.
2. **Labels** (Stage 2): ESA WorldCover cropland mask exported at the same
   10m/EPSG:4326 grid as the imagery, for direct pixel alignment.
3. **Preprocessing** (Stage 3): L2A is already Bottom-of-Atmosphere
   corrected (Sen2Cor), so no additional atmospheric correction step was
   needed. Tiles were chipped into 64×64 patches; NDVI/NDWI/EVI were
   computed and stacked as extra channels (13 total per patch); patches
   were **split by spatial grid block, not randomly**, so spatially
   adjacent (and therefore correlated) patches can't leak across
   train/val/test this avoids inflated accuracy from spatial
   autocorrelation.
4. **Modeling** (Stage 4): three approaches trained on the same patches and
   evaluated on the same held-out test set with the same metrics.
5. **Comparison** (Stage 5): accuracy, macro F1, and per-class F1
   consolidated into a table and chart.

### Why NDVI / NDWI / EVI 

- **NDVI** `(NIR-Red)/(NIR+Red)`: chlorophyll absorbs red light while leaf
  mesophyll strongly reflects NIR, so NDVI cleanly separates vegetated
  cropland from bare soil, built-up, or fallow land, and its seasonal
  trajectory tracks crop growth stages.
- **NDWI** `(Green-NIR)/(Green+NIR)`: sensitive to canopy/leaf water
  content helps distinguish irrigated or flooded (paddy) cropland from
  dry vegetation or senescent/harvested fields.
- **EVI**: NDVI-like but corrects for soil background and atmospheric
  scattering via the blue band, and stays sensitive at high biomass where
  NDVI saturates which is useful for resolving growth-stage differences within
  already-vegetated fields.

## Models compared

| Model | Approach | Notes |
|---|---|---|
| Random Forest | scikit-learn, hand-crafted features (per-band mean/std across all 13 channels) | Classical baseline |
| UNet | `segmentation-models-pytorch`, ResNet18 encoder, trained from scratch | Pixel-wise segmentation, aggregated to patch level for fair comparison |
| ViT Foundation Model | TorchGeo `ViTSmall16_Weights.SENTINEL2_ALL_MAE` (SSL4EO-S12), fine-tuned | **Substitute for SatMAE/Prithvi** — see note below |

**Foundation model substitution note:** SatMAE's checkpoints are hosted on
the authors' project page, not pip/torchgeo-installable; Prithvi requires a
HuggingFace-gated download and remapping to HLS's band set. Both add setup
friction disproportionate to a Colab free-tier prototype. TorchGeo ships an
ungated, directly-loadable Sentinel-2 encoder pretrained with the same
Masked-Autoencoder self-supervised objective SatMAE uses (via SSL4EO-S12,
Wang et al. 2022) used here as the closest viable, actually-runnable
substitute. Two further approximations from this substitution, both flagged
in the code: (1) the pretrained encoder expects 13 Sentinel-2 bands and our
export has 10 (B1/B9/B10 dropped to save export size L2A doesn't include
B10 at all), so the 3 missing channels are zero-filled; (2) the encoder
expects 224×224 input, so our 64×64 patches are bilinearly upsampled before
being fed in.



## Future Scope

- Multi-class crop-type labels (field-survey or a verified dataset with
  confirmed AOI overlap), extending all three models beyond binary
  cropland detection.
- Larger AOIs and more seasonal windows for stronger phenological signal.
- Native-resolution input to the foundation model instead of upsampling
  64×64 patches to 224×224 (e.g. re-chipping at 224×224 directly, at the
  cost of coarser spatial granularity per training example).
- Hyperparameter search for the Random Forest and UNet (this run uses
  reasonable defaults, not a tuned configuration).
- Full 13-band exports (adding B1/B9/B10) so the foundation model doesn't
  need zero-filled channels.
