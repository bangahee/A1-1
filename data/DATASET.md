# Dataset provenance

- Source: UCI Machine Learning Repository, **Online Retail**
- Creator: Daqing Chen
- DOI: <https://doi.org/10.24432/C5BW33>
- Dataset page: <https://archive.ics.uci.edu/dataset/352/online%2Bretail>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Original size: 541,909 transaction rows, 8 source columns
- Local sample: 25,000 rows selected with NumPy/Pandas seed `42`

The original data describes transactions from 2010-12-01 through 2011-12-09
for a UK-based non-store retailer. The local CSV retains source transaction
fields and adds reproducible derived fields. `amount` is `quantity *
unit_price`. `product_image` is an 8x8 grayscale-like numeric array generated
deterministically from `stock_code`; it is an educational array feature, not a
photograph supplied by UCI. This distinction prevents fabricated image
provenance while satisfying the mission's multimodal array exercise.

`source_evidence.png` is a direct capture of the official dataset page.

Suggested citation:

> Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5BW33

