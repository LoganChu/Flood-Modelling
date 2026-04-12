# Data Quality Notes

## Summary

All 6 datasets have been processed and visualized. One dataset (`enoM30(3).csv`) shows anomalous GPS values that merit investigation.

---

## Datasets Overview

| Dataset | Samples | E-W Extent | N-S Extent | Altitude Range | Status |
|---------|---------|-----------|-----------|----------------|--------|
| enoFeb16th | 3,553 | 114 m | 337 m | 65.7 m | ✓ Clean |
| enoFeb16th_smoothed | 3,553 | 114 m | 337 m | 65.7 m | ✓ Clean |
| enoM30(1) | 7,284 | 129 m | 344 m | 64.0 m | ✓ Clean |
| enoM30(2) | 8,988 | 122 m | 324 m | 61.7 m | ✓ Clean |
| **enoM30(3)** | 7,076 | **14,200 km** | **3,945 km** | 45.1 m | ⚠️ **ANOMALY** |
| enoM30(4) | 9,532 | 126 m | 347 m | 196 m | ✓ Clean |

---

## Data Quality Issues

### enoM30(3).csv — Anomalous GPS Data

**Issue:** Extreme geographic extent indicates GPS dropout or coordinate system mismatch.

**Evidence:**
- East-West extent: 14,200 km (should be ~100–150 m for a river drifter)
- North-South extent: 3,945 km (should be ~300–400 m)
- Altitude range: 45.1 m (reasonable)

**Likely causes:**
1. **GPS column swap** — Lat/Lon coordinates may be transposed or misaligned
2. **Coordinate system mismatch** — Data recorded in a different reference frame
3. **Sensor malfunction** — GPS receiver produced invalid readings during portions of this deployment
4. **Data concatenation error** — Multiple datasets merged with conflicting coordinate systems

**Impact:**
- 2D and 3D visualizations show distorted (nearly useless) trajectories
- GeoConverter's `pyproj` coordinate transform may fail or produce invalid results on this data
- Not suitable for EKF state estimation or physics validation without preprocessing

**Recommendation:**
- **Investigate the raw CSV** — Check for header misalignment, missing columns, or sensor logs from the deployment
- **Cross-reference with field notes** — Confirm deployment date/location and sensor configuration
- **Consider excluding** from Phase 3–4 analysis (EKF, metrics, physics validation) until root cause is identified
- **If recoverable**, consider data cleaning script to swap/correct coordinate columns

---

## Good Datasets

### enoFeb16th (Original deployment)
- **Status**: ✓ Baseline reference
- **Characteristics**: Balanced trajectory with good speed variation (0–2.45 m/s)
- **Use case**: Primary test dataset for Phase 1–2 validation
- **Note**: ~36-minute deployment with clear start/end

### enoFeb16th_smoothed (Pre-filtered version)
- **Status**: ✓ Comparison baseline
- **Characteristics**: Same trajectory as raw, but pre-smoothed for comparison
- **Use case**: Benchmark for EKF filter improvement metrics (Phase 3)
- **Note**: Allows RMSE comparison: raw → EKF vs. raw → smoothed

### enoM30(1), enoM30(2), enoM30(4) (Extended deployments)
- **Status**: ✓ Rich trajectories
- **Characteristics**: 7k–9.5k samples, longer durations, complex meanders
- **Use case**: Phase 3–4 testing with diverse river conditions
- **enoM30(4)**: Interesting altitude variation (196 m) — different deployment depth?

---

## Recommendations

### For Phase 1–2 (Current Status)
- Use **enoFeb16th** or **enoFeb16th_smoothed** for primary validation
- Use **enoM30(1), (2), (4)** for extended test coverage
- **Exclude enoM30(3)** until investigated

### For Phase 3–4 (EKF & Metrics)
- Build test suite using clean datasets:
  - Short baseline: enoFeb16th (quick validation)
  - Extended baseline: enoM30(1), (2), (4)
- Document expected RMSE/filter improvement ranges per dataset
- Compare filter performance across different trajectory lengths

### For Data QA
- Add automated checks in `data_loader.py`:
  - Flag lat/lon extent > 1 km (sanity check)
  - Log GPS dropout statistics per deployment
  - Validate coordinate system consistency (e.g., check that first Lat/Lon is within expected bounds)

---

## Visualization Quality

All visualizations (2D PNG + 3D HTML) have been generated for all 6 datasets:

- **Total size**: 36 MB (7 PNG files @ ~100–200 KB each, 7 HTML files @ ~4.8–5.1 MB each)
- **enoM30(3)**: Visualizations show the anomaly clearly — the 2D PNG is notably smaller (69 KB) due to plot axis distortion

**Viewing:**
- PNG files: Open in any image viewer
- HTML files: Open in web browser (Chrome, Firefox, Safari, Edge)
- All visualizations respect the data cleaning pipeline (dropout removal, time normalization)
