# Data Sources, Open Licenses & Verified Citations

> **Legal Notice:** Research Software — NOT A MEDICAL DEVICE. All datasets generated or staged by `quantnuc-inverse` are strictly for scientific research, phantom benchmarking, and algorithmic verification.

## 1. Zero-Plagiarism & Data Provenance Policy

Every dataset staged or synthesized by `quantnuc-inverse` carries an explicit provenance tag (`SYNTHETIC_NEMA_NU2_2018_DRO`, `UPSTREAM_PYDICOM_OPEN_TEST_DATA`, or `ANALYTICAL_PHYSICS_MODEL`) inside every `ProvenanceRecord` and SHA-256 `MANIFEST.json`.

- **Synthetic NEMA NU 2-2018 Digital Reference Object (DRO)**: Constructed from first principles per NEMA NU 2-2018 IEC Body Phantom geometry (6 hot fillable spheres: 10, 13, 17, 22, 28, and 37 mm inner diameters on a 57.2 mm ring radius; 4:1 sphere-to-background ratio; central 50 mm lung insert; wall thickness $\le 1.0\text{ mm}$) via `scripts/fetch_public_nucmed_sample.py`. Licensed Apache-2.0.
- **Bundled Public CT Sample**: `pydicom.examples.ct` staged as `data/sample_nema_dro/pydicom_public_ct_sample.dcm` under the upstream `pydicom` MIT License (verbatim copy included in `data/sample_nema_dro/LICENSE` with SHA-256 checksum in `data/sample_nema_dro/MANIFEST.json`).
- **Linear Attenuation Coefficients**: Standard water/soft-tissue and inflated-lung linear attenuation values at 140, 208, 364, and 511 keV derived from public-domain NIST XCOM / ICRU-44 tables.
- **No proprietary vendor software, clinical patient PHI, or restricted closed-license datasets** are included anywhere in this repository.

## 2. Verified Literature References

| Topic | Citation | Identifier / License |
|---|---|---|
| EARL2 Multi-Center Harmonization (89Zr & PET/CT) | Kaalep A, Huisman MC, Sera T, Vugts D, Boellaard R. *Feasibility of PET/CT system performance harmonisation for quantitative multicentre 89Zr studies.* EJNMMI Physics (2018) 5:14 | DOI: [`10.1186/s40658-018-0226-7`](https://doi.org/10.1186/s40658-018-0226-7) (CC BY 4.0) |
| EARL2 Updated FDG-PET/CT Standards | Kaalep A, Sera T, Rijnsdorp S, et al. *Feasibility of state of the art PET/CT systems performance harmonisation.* Eur J Nucl Med Mol Imaging (2018) 45:1344–1361 | DOI: [`10.1007/s00259-018-3977-4`](https://doi.org/10.1007/s00259-018-3977-4) (CC BY 4.0) |
| EARL1 First 200 Systems Summary | Kaalep A, Sera T, Oyen W, et al. *EANM/EARL FDG-PET/CT accreditation - summary results from the first 200 accredited imaging systems.* Eur J Nucl Med Mol Imaging (2018) 45:412–422 | DOI: [`10.1007/s00259-017-3853-7`](https://doi.org/10.1007/s00259-017-3853-7) |
| EANM FDG-PET/CT Tumour Guidelines v2.0 | Boellaard R, Delgado-Bolton R, Oyen WJG, et al. *FDG PET/CT: EANM procedure guidelines for tumour imaging: version 2.0.* Eur J Nucl Med Mol Imaging (2015) 42:328–354 | DOI: [`10.1007/s00259-014-2961-x`](https://doi.org/10.1007/s00259-014-2961-x) |
| Geometric Transfer Matrix (GTM) PVC | Rousset OG, Ma Y, Evans AC. *Correction for partial volume effects in PET: principle and validation.* J Nucl Med (1998) 39(5):904–911 | PMID: `9591599` |
| Voxel-Wise PVC (Müller–Gärtner) | Müller-Gärtner HW, Links JM, Prince JL, et al. *Measurement of radiotracer concentration in brain gray matter using positron emission tomography: MRI-based correction for partial volume effects.* J Cereb Blood Flow Metab (1992) 12(4):571–583 | DOI: [`10.1038/jcbfm.1992.81`](https://doi.org/10.1038/jcbfm.1992.81) |
| Quantitative SPECT Dosimetry (MIRD 23) | Dewaraja YK, Frey EC, Sgouros G, et al. *MIRD Pamphlet No. 23: Quantitative SPECT for Patient-Specific 3-Dimensional Dosimetry in Internal Radionuclide Therapy.* J Nucl Med (2012) 53(8):1310–1325 | DOI: [`10.2967/jnumed.111.100123`](https://doi.org/10.2967/jnumed.111.100123) |
| Quantitative Lu-177 SPECT (MIRD 26) | Ljungberg M, Celler A, Konijnenberg MW, et al. *MIRD Pamphlet No. 26: Joint EANM/MIRD Guidelines for Quantitative 177Lu SPECT Applied for Dosimetry of Radiopharmaceutical Therapy.* J Nucl Med (2016) 57(1):151–162 | DOI: [`10.2967/jnumed.115.159012`](https://doi.org/10.2967/jnumed.115.159012) |
| Global Sobol' Sensitivity (Saltelli–Jansen) | Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S. *Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index.* Comput Phys Commun (2010) 181(2):259–270 | DOI: [`10.1016/j.cpc.2009.09.018`](https://doi.org/10.1016/j.cpc.2009.09.018) |
| Region-Based Voxel-Wise (RBV) PVC | Thomas BA, Erlandsson K, Modat M, et al. *The importance of appropriate partial volume correction for PET quantification in Alzheimer's disease.* Eur J Nucl Med Mol Imaging (2011) 38:1104–1119 | DOI: [`10.1007/s00259-011-1745-9`](https://doi.org/10.1007/s00259-011-1745-9) |
| Bilinear CT HU-to-$\mu(E)$ AC Transform | Carney JPJ, Townsend DW, Rappoport V, Bendriem B. *Method for transforming CT images for attenuation correction in PET/CT imaging.* Med Phys (2006) 33(4):976–983 | DOI: [`10.1118/1.2174132`](https://doi.org/10.1118/1.2174132) |
| PET Positron-Range Physics | Conti M, Eriksson L. *Physics of pure and non-pure positron emitters for PET: a review and a discussion.* EJNMMI Physics (2016) 3:8 | DOI: [`10.1186/s40658-016-0144-5`](https://doi.org/10.1186/s40658-016-0144-5) (CC BY 4.0) |
| NEMA IEC Body Phantom Standard | National Electrical Manufacturers Association (NEMA). *NEMA Standards Publication NU 2-2018: Performance Measurements of Positron Emission Tomographs (PET).* (2018) | NEMA NU 2-2018 |

## 3. BibTeX Entries

```bibtex
@article{kaalep2018ejnmmiphys,
  author  = {Kaalep, Andres and Huisman, Marc C. and Sera, Terez and Vugts, Danielle and Boellaard, Ronald},
  title   = {Feasibility of {PET/CT} system performance harmonisation for quantitative multicentre $^{89}${Zr} studies},
  journal = {EJNMMI Physics},
  year    = {2018},
  volume  = {5},
  pages   = {14},
  doi     = {10.1186/s40658-018-0226-7}
}

@article{kaalep2018earl2,
  author  = {Kaalep, Andres and Sera, Terez and Rijnsdorp, S. and Yaqub, Maqsood and Talsma, A. and Lodge, Martin A. and Boellaard, Ronald},
  title   = {Feasibility of state of the art {PET/CT} systems performance harmonisation},
  journal = {European Journal of Nuclear Medicine and Molecular Imaging},
  year    = {2018},
  volume  = {45},
  pages   = {1344--1361},
  doi     = {10.1007/s00259-018-3977-4}
}

@article{boellaard2015eanm,
  author  = {Boellaard, Ronald and Delgado-Bolton, Roberto and Oyen, Wim J. G. and Giammarile, Francesco and Tatsch, Klaus and Eschner, Wolfgang and Verzijlbergen, Fred J. and Barrington, Sally F. and Pike, Lucy C. and Weber, Wolfgang A. and Stroobants, Sigrid and Delbeke, Dominique and Donohoe, Kevin J. and Holbrook, Scott and Graham, Michael M. and Testanera, Giorgio and Hoekstra, Otto S. and Zijlstra, Josee and Visser, Eric and Hoekstra, Corneline J. and Pruim, Jan and Willemsen, Antoon and Arends, Bertjan and Kotzerke, Joerg and Bockisch, Andreas and Beyer, Thomas and Chiti, Arturo and Krause, Bernd J.},
  title   = {{FDG PET/CT}: {EANM} procedure guidelines for tumour imaging: version 2.0},
  journal = {European Journal of Nuclear Medicine and Molecular Imaging},
  year    = {2015},
  volume  = {42},
  pages   = {328--354},
  doi     = {10.1007/s00259-014-2961-x}
}

@article{rousset1998gtm,
  author  = {Rousset, Olivier G. and Ma, Yilong and Evans, Alan C.},
  title   = {Correction for partial volume effects in {PET}: principle and validation},
  journal = {Journal of Nuclear Medicine},
  year    = {1998},
  volume  = {39},
  number  = {5},
  pages   = {904--911},
  pmid    = {9591599}
}

@article{mullergartner1992pvc,
  author  = {M{\"u}ller-G{\"a}rtner, Hans W. and Links, Jonathan M. and Prince, Jerry L. and Bryan, R. Nick and McVeigh, Elliot and Leal, Jeffrey P. and Davatzikos, Christos and Frost, J. James},
  title   = {Measurement of radiotracer concentration in brain gray matter using positron emission tomography: {MRI}-based correction for partial volume effects},
  journal = {Journal of Cerebral Blood Flow \& Metabolism},
  year    = {1992},
  volume  = {12},
  number  = {4},
  pages   = {571--583},
  doi     = {10.1038/jcbfm.1992.81}
}

@article{dewaraja2012mird23,
  author  = {Dewaraja, Yuni K. and Frey, Eric C. and Sgouros, George and Brill, A. Bertrand and Roberson, Peter and Zanzonico, Pat B. and Ljungberg, Michael},
  title   = {{MIRD} Pamphlet No. 23: Quantitative {SPECT} for Patient-Specific 3-Dimensional Dosimetry in Internal Radionuclide Therapy},
  journal = {Journal of Nuclear Medicine},
  year    = {2012},
  volume  = {53},
  number  = {8},
  pages   = {1310--1325},
  doi     = {10.2967/jnumed.111.100123}
}

@article{ljungberg2016mird26,
  author  = {Ljungberg, Michael and Celler, Anna and Konijnenberg, Mark W. and Eckerman, Keith F. and Dewaraja, Yuni K. and Sj{\"o}green-Gleisner, Katarina},
  title   = {{MIRD} Pamphlet No. 26: Joint {EANM/MIRD} Guidelines for Quantitative $^{177}${Lu} {SPECT} Applied for Dosimetry of Radiopharmaceutical Therapy},
  journal = {Journal of Nuclear Medicine},
  year    = {2016},
  volume  = {57},
  number  = {1},
  pages   = {151--162},
  doi     = {10.2967/jnumed.115.159012}
}

@article{saltelli2010sobol,
  author  = {Saltelli, Andrea and Annoni, Paola and Azzini, Ivano and Campolongo, Francesca and Ratto, Marco and Tarantola, Stefano},
  title   = {Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index},
  journal = {Computer Physics Communications},
  year    = {2010},
  volume  = {181},
  number  = {2},
  pages   = {259--270},
  doi     = {10.1016/j.cpc.2009.09.018}
}
```
