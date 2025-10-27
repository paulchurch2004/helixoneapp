# 🏦 BIS API Migration Notes

**Date**: 2025-10-22
**Status**: REQUIRES REFACTORING ⚠️

## Problem

BIS (Bank for International Settlements) migrated their API in 2024-2025:
- **Old API**: `https://data.bis.org/api/v1/` ❌
- **New API**: `https://stats.bis.org/api/v1/` ✅

## Changes Required

### 1. Base URL ✅ FIXED
```python
# OLD
BIS_BASE_URL = "https://data.bis.org/api/v1"

# NEW
BIS_BASE_URL = "https://stats.bis.org/api/v1"
```

### 2. Headers ✅ FIXED
```python
# Must use SDMX 2.1 headers
headers = {
    'Accept': 'application/vnd.sdmx.data+json;version=1.0.0',
    'User-Agent': 'HelixOne/1.0 (Financial Data Platform)'
}
```

### 3. Dataset Names ⚠️ TODO
Dataflow names changed from `WEBSTATS_XXX_DATAFLOW` to `WS_XXX`:

| Old Name | New Name | Description |
|----------|----------|-------------|
| `WEBSTATS_CREDIT_DATAFLOW` | `WS_CREDIT_GAP` | Credit-to-GDP gaps |
| `WEBSTATS_LONG_DATAFLOW` | `WS_TC` | Total credit (long series) |
| `WEBSTATS_DEBTSEC_DATAFLOW` | `WS_DEBT_SEC2_PUB` | Debt securities |
| `WEBSTATS_EER_DATAFLOW` | `WS_EER` | Effective exchange rates |
| `WEBSTATS_RPPI_DATAFLOW` | `WS_SPP` | Residential property prices |
| `WEBSTATS_OTC_DERIV_DATAFLOW` | `WS_OTC_DERIV2` | OTC derivatives |
| `WEBSTATS_CBPOL_DATAFLOW` | `WS_CBPOL` | Central bank policy rates |
| `WEBSTATS_GLI_DATAFLOW` | `WS_GLI` | Global liquidity |
| `WEBSTATS_CBS_DATAFLOW` | `WS_CBS_PUB` | Consolidated banking stats |

### 4. Key Structure ⚠️ TODO
The dimension order and codes changed. Example for EER:

**Old format**:
```
M.{COUNTRY}.{TYPE}.{BASKET}
M.DE.R.N  # Monthly, Germany, Real, Narrow
```

**New format**:
```
M.{TYPE}.{BASKET}.{COUNTRY}
M.R.B.DE  # Monthly, Real, Broad, Germany
```

**Working example**:
```bash
curl -H "Accept: application/vnd.sdmx.data+json;version=1.0.0" \
  "https://stats.bis.org/api/v1/data/WS_EER/M.N.B.US?startPeriod=2024-01&endPeriod=2024-12"
```

## Next Steps

### Option A: Full Refactoring (3-4 hours)
- Update all 9 methods with new dataflow names
- Fix all key structures for each dataset
- Update tests
- Document new parameter formats

### Option B: Gradual Migration
- Mark BIS as "Requires Refactoring" in status
- Focus on fixing easier sources (SEC Edgar, IMF)
- Come back to BIS later with full refactor

### Option C: Use Alternative
- BIS data is also available via:
  - FRED (some overlap)
  - IMF (some overlap)
  - ECB (European data)
  - Direct web interface: https://data.bis.org

## Resources

- API Documentation: https://stats.bis.org/api-doc/v2/
- Dataflow List: `https://stats.bis.org/api/v1/dataflow`
- SDMX 2.1 Spec: https://www.bis.org/statistics/sdmx.htm

## Recommendation

**Proceed with Option B**: Mark as refactoring needed and prioritize sources that can be fixed quickly (SEC Edgar, IMF). BIS data overlaps significantly with other sources we already have working (FRED, ECB, World Bank).
