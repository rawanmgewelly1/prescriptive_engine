# Local acceptance results for updated multipart delivery

After adding the user-supplied Solar CatBoost model, the full multipart delivery was extracted into a fresh directory and its Wind CSV restored from seven individually downloaded ZIPs.

- `python RESTORE_WIND_DATA.py`: PASS. Original Wind CSV restored with verified SHA-256.
- `python tools/verify_manifest.py`: PASS, 129/129 files verified at packaging time (manifest regenerated after this report).
- `python -m unittest discover -s tests -q`: PASS, 159 tests, one skipped.
- `python run_pipeline.py --scenario configs/scenarios/RESEARCH_DEMO_200KW.json`: `RESEARCH_CONDITIONAL_RESULT_NOT_PHYSICAL_VALIDATION`.
- The supplied Solar CatBoost artifact has eight training feature names matching `config.json`; 1,745 Solar predictions were generated in a direct local inference test. Of these, 471 were negative and require diagnosis; they are not physically valid available power.

These checks demonstrate that the research software operates under the supplied assumptions; they do not validate chronological Solar/Wind coupling, physical hardware integration, safety, or model generalization. Serialized model library version warnings remain relevant.
