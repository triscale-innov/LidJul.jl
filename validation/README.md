# Scientific validation

Load `cavity_reference.jl` from the benchmark environment and call
`cavity_validation(output="validation/results.toml")`. See the Documenter
validation page for the numerical method, source, tolerances and limitations.

The reference CSV contains the Re=100 centerline samples from Tables I and II of
Ghia, Ghia and Shin (1982), DOI 10.1016/0021-9991(82)90058-4. The data were
checked against the original article tables; numeric OCR spacing and minus-sign
artifacts were normalized. No downloads are required to run the comparison.
