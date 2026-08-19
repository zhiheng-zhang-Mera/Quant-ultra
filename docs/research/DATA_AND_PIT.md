# Data and PIT Methodology

Each core dataset is identified by source, URI, licence/terms label, download time, coverage, raw file, cleaned file, transformations, anomalies and optional predecessor/source-replacement history. Raw and cleaned SHA-256 values produce a content-addressed dataset version.

The PIT contract requires publish/event time and ingest/availability time where applicable. Historical replay must supply `as_of`; a later observation cannot affect an earlier feature prefix. Missing required histories or invalid hashes stop the experiment instead of silently substituting data.

Data manifests are checked at consumption time. A changed source file, cleaned file, transformation description or provenance record changes the version. Hash-chained provenance events make later alteration detectable.

Vendor-adjusted market history can be revised after download. A manifest proves which bytes were used, not that the vendor's bytes were economically correct. Data-quality evidence and source licensing remain separate responsibilities.
