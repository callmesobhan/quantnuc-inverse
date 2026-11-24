# quantnuc-inverse: Mathematical & Engineering Specification

> **Notice:** Research Software — NOT A MEDICAL DEVICE.

## 1. Physics Intermediate Representation (`src/quantnuc_inverse/ir.py`)

All inputs and outputs are strictly typed, immutable Pydantic v2 models (`ConfigDict(frozen=True)`) enforcing physical bounds and unit conventions:

- **`ScannerSpec`**: Transverse & axial intrinsic FWHM ($\text{mm}$), anisotropic voxel spacing $(dz, dy, dx)$, off-axis parallax / depth-dependent CDR broadening ($\text{mm}/10\text{ cm}$), TOF resolution ($\text{ps}$), PSF resolution modeling flag, dimensionless Gibbs ringing strength $\gamma \in [0, 0.6]$, sensitivity ($\text{cps/kBq}$), and isotope.
- **`ReconConfig`**: Algorithm (`OSEM`, `PSF_OSEM`, `TOF_PSF_OSEM`, `BSREM`), iterations, subsets, post-reconstruction filter (`gaussian` or `butterworth`), Butterworth order $n$, BSREM penalty $\beta$, and convergence constant $k$.
- **`AbstentionError`**: Hard quantitative validity exception carrying a structured `AbstentionCode` (`UNCALIBRATED_SUV_METADATA`, `SEVERE_CT_AC_MISMATCH`, `SUB_NYQUIST_LESION_UNRECOVERABLE`, `EARL_CORRIDOR_INFEASIBLE`, `MISSING_DECAY_CORRECTION`) and JSON-serializable audit payload.

## 2. 3D Forward Image Formation & PSF-Gibbs Operator (`forward_image_formation.py`)

### 2.1 Spatially Variant Effective PSF / CDR
At radial offset $r$ (mm):
$$\text{FWHM}_{\text{sys},xy}(r) = \sqrt{\text{FWHM}_{xy}^2 + \left(\alpha_r \frac{r}{100}\right)^2}$$
Resolution modeling (`PSF_OSEM`, `TOF_PSF_OSEM`, `BSREM`) sharpens intrinsic PSF while incomplete OSEM iteration convergence ($c = 1 - e^{-k \cdot N_{\text{iter}} N_{\text{sub}}}$) and BSREM regularization ($\beta$) add quadrature smoothing.

### 2.2 Activity-Conserving PSF-Gibbs Edge Ringing Kernel
To reproduce PSF-induced Gibbs ringing ("SUVmax inflation") while strictly preserving linearity and total activity conservation, the forward system operator $\mathcal{H}$ combines the base 3D anisotropic Gaussian response $G_{\sigma_{\text{pre}}}$ with a zero-integral Difference-of-Gaussians (Mexican-hat) edge-enhancement kernel:
$$\mathcal{H}[I] = F_{\text{post}} * \left[ G_{\sigma_{\text{pre}}} * I + 2.75\,\gamma_{\text{eff}} \left( G_{0.78\,\sigma_{\text{pre}}} * I - G_{2.15\,\sigma_{\text{pre}}} * I \right) \right]$$
Because $\int G_{\sigma}\,d\mathbf{x} = 1$, $\int (G_{0.78\sigma} - G_{2.15\sigma})\,d\mathbf{x} = 0$, guaranteeing:
1. Exact linearity: $\mathcal{H}[\alpha I_1 + \beta I_2] = \alpha \mathcal{H}[I_1] + \beta \mathcal{H}[I_2]$.
2. Exact preservation of uniform fields: $\mathcal{H}[c] = c$.
3. Exact total activity conservation under conservative boundaries.
4. Smooth physical suppression of Gibbs overshoot as post-reconstruction filter $F_{\text{post}}$ FWHM increases.

For `filter_type="butterworth"`, $F_{\text{post}}$ is applied in 3D Fourier space using the exact $n$-th order amplitude transfer function:
$$H_{\text{BW}}(f) = \frac{1}{\sqrt{1 + (f / f_c)^{2n}}}, \qquad f_c = \frac{0.44}{\text{FWHM}_{\text{eq}}}$$

### 2.3 Exact CT-AC Respiratory Mismatch Kernel
When CT breathing phase shifts by $\Delta z$ relative to PET/SPECT across a curved diaphragmatic dome of radius $R_{\text{dome}}$, the mismatched tissue chord length $L_{\text{eff}}(\Delta z)$ induces an exact exponential attenuation bias:
$$\frac{C_{\text{recon}}}{C_{\text{true}}} = \exp\!\left(\int_{\text{LOR}} \Delta\mu(s)\,ds\right)$$
- $\Delta z > 0$ (CT inspiration shifts liver inferiorly, assigning lung $\mu$ to liver dome): $\Delta\mu < 0 \Rightarrow \text{FALSE\_COLD}$ underestimation.
- $\Delta z < 0$ (CT expiration shifts liver superiorly, assigning liver $\mu$ to basal lung): $\Delta\mu > 0 \Rightarrow \text{FALSE\_HOT}$ overestimation.
- Exceeding $\pm 20\%$ triggers `AbstentionCode.SEVERE_CT_AC_MISMATCH`.

## 3. Partial-Volume Inverse Recovery (`pve_inverse_solver.py`)

### 3.1 Rousset Geometric Transfer Matrix (GTM)
For an $M$-region partition $\{R_1, \dots, R_M\}$, using the exact forward operator $\mathcal{H}$:
$$W_{ij} = \frac{1}{|R_i|} \sum_{\mathbf{x} \in R_i} (\mathcal{H} R_j)(\mathbf{x}), \qquad \hat{\mathbf{T}} = \mathbf{W}^{-1} \mathbf{t}_{\text{obs}}$$
**Formal Invariant:** $\|\mathbf{W} \hat{\mathbf{T}} - \mathbf{t}_{\text{obs}}\|_\infty \le 10^{-6}$ (numerically $\sim 10^{-16}$ in float64) and total activity conservation error $\le 10^{-6}$ on closed partitions.

### 3.2 Müller–Gärtner (MG) Voxel-Wise PVC & Bayesian Regularized Inversion
- **MG Voxel-Wise**: $I_{\text{MG}}(\mathbf{x}) = [I_{\text{obs}}(\mathbf{x}) - C_{\text{bg}} (\mathcal{H} R_{\text{bg}})(\mathbf{x})] / (\mathcal{H} R_{\text{target}})(\mathbf{x})$ for $(\mathcal{H} R_{\text{target}})(\mathbf{x}) \ge \tau$.
- **Bayesian MAP / Tikhonov**: $(\mathbf{W}^\top \mathbf{W} + \lambda \mathbf{I})\hat{\mathbf{T}} = \mathbf{W}^\top \mathbf{t}_{\text{obs}} + \lambda \mathbf{T}_{\text{prior}}$ with Monte Carlo propagation of PSF FWHM, ROI boundary delineation ($\pm 0.5$ voxel), Gibbs amplitude, and volume-scaled Poisson counting noise yielding calibrated `[SUV_p05, nominal, SUV_p95]` intervals.

## 4. Multi-Scanner Harmonization Optimizer (`earl_harmonizer.py`)

Solves the constrained minimal-smoothing problem over NEMA NU 2-2018 6-sphere RC corridors (`EARL1`, `EARL2`, `QIBA_FDG_PET`, `MIRD23_SPECT`):
$$\min_{\text{FWHM}_{\text{post}} \ge 0,\; N_{\text{iter}}} \text{FWHM}_{\text{post}} \quad \text{s.t.} \quad \text{RC}_m^{\text{lower}}(d) \le \text{RC}_m(d) \le \text{RC}_m^{\text{upper}}(d)$$
for $m \in \{\text{mean}, \text{max}, \text{peak}\}$ and $d \in \{10, 13, 17, 22, 28, 37\}\,\text{mm}$, using cached 3D anti-aliased sphere masks, bisection on the minimal feasible boundary, and bounded Brent refinement (`scipy.optimize.minimize_scalar`). Infeasible scanners trigger `AbstentionCode.EARL_CORRIDOR_INFEASIBLE`.

## 5. Quantitative DICOM Calibration (`dicom_io.py`)

Verifies `Modality` (`PT`/`NM`), `DecayCorrection` (`START`/`ADMIN`), `Units == "BQML"`, `RescaleSlope`, `PatientWeight`, and `RadiopharmaceuticalInformationSequence`, computing:
$$D_{\text{corr}} = D_0 \cdot 2^{-\Delta t / T_{1/2}}, \qquad \text{SUV}_{\text{bw}} = \frac{C_{\text{Bq/mL}} \cdot 1000\,W_{\text{kg}}}{D_{\text{corr}}}, \qquad \text{SUV}_{\text{lbm}} = \frac{C_{\text{Bq/mL}} \cdot 1000\,\text{LBM}_{\text{kg}}}{D_{\text{corr}}}$$
using Janmahasatian lean body mass (PERCIST 1.0 / EANM v2.0).

## 6. Global Sobol' Sensitivity (`sobol_sensitivity.py`)

Uses scrambled QMC Sobol' sequences ($N = 2^m$ base samples, $N(k+2)$ evaluations) with Jansen (1999) / Saltelli (2010) low-variance variance-based estimators:
$$S_{1,i} = 1 - \frac{\frac{1}{2N}\sum_{j=1}^N \left(y_{B,j} - y_{A_B^{(i)},j}\right)^2}{\text{Var}(Y)}, \qquad S_{T,i} = \frac{\frac{1}{2N}\sum_{j=1}^N \left(y_{A,j} - y_{A_B^{(i)},j}\right)^2}{\text{Var}(Y)}$$
with bootstrap 95% confidence intervals across PSF FWHM, ROI delineation error ($\pm 1$ voxel), Gibbs ringing, and CT-AC respiratory phase shift.

## 7. Frontier Agent Harness, CLI & MCP Server (`harness/`, `cli.py`)

- `QuantitativeImagingForensicsAgent`: Performs Shapley-exact logarithmic attribution separating observed longitudinal $\Delta\text{SUV}$ into PSF-Gibbs, PVE/resolution, CT-AC mismatch, and true biological change; runs `CriticVerifierAudit`; exports Markdown + SHA-256 JSON dossiers.
- `mcp_server.py`: Line-delimited JSON-RPC 2.0 stdio server exposing 9 provenance-gated tools (`quantnuc_forward_nema_rc`, `quantnuc_invert_pve_activity`, `quantnuc_synthesize_earl_filter`, `quantnuc_audit_ct_ac_mismatch`, `quantnuc_diagnose_suv_shift`, `quantnuc_run_sobol_sensitivity`, `quantnuc_compute_theranostic_dose`, `quantnuc_audit_3d_ct_ac_volume`, `quantnuc_generate_physics_dossier`).
- `cli.py`: Ergonomic `quantnuc` CLI exposing `harmonize`, `pvc`, `audit-ac`, `dose`, and `dossier`.

## 8. v0.2.0 Physics & Theranostics Extensions

- **Isotope Positron-Range Blur**: $\text{FWHM}_{\text{pos}}(\text{isotope}, \rho) = \text{FWHM}_{\text{water}} / \max(\rho, 0.15)$ added in quadrature to intrinsic detector PSF for `F-18` (0.54 mm), `Zr-89` (1.27 mm), `Ga-68` (2.90 mm), and `I-124` (3.40 mm) (Conti & Eriksson 2016).
- **3D Bilinear HU-to-$\mu(E)$ Ray-Traced CT-AC**: Piecewise Carney et al. (2006) 120 kVp transform mapped to 511/364/208/140 keV + paired 3D CT-to-PET affine resampling (`resample_ct_to_pet_grid`) and multi-slice `ImagePositionPatient`-sorted DICOM series ingestion.
- **Region-Based Voxel-Wise (RBV) PVC**: Thomas et al. (2011) GTM + Yang ratio correction $I_{\text{RBV}}(\mathbf{x}) = I_{\text{obs}}(\mathbf{x}) \cdot S(\mathbf{x}) / (\mathcal{H}S)(\mathbf{x})$ for heterogeneous necrotic-core + viable-rim lesions.
- **Noise-Constrained EARL2 Pareto Frontier**: Jointly evaluates RC corridor compliance, EANM background noise $\text{COV}_{\text{bg}} \le 15\%$, and 13 mm sphere CNR across `(iterations, FWHM)`.
- **MIRD-23/26 Theranostic Absorbed-Dose Engine (`theranostics_dosimetry.py`)**: Multi-timepoint mono/bi-exponential TIAC fitting with `INSUFFICIENT_TIMEPOINTS_FOR_TIAC` abstention gate, finite-sphere beta/gamma MIRD S-factors (`Lu-177`, `I-131`, `Y-90`), and exact PVE `[p05, nominal, p95]` propagation into tumor absorbed dose (Gy).

## 9. v0.3.0 Flagship Inverse & Protocol-Design Capabilities

- **$N=1$ Retrospective Cross-Scanner Translation Operator (`scanner_translator.py`)**:
  - Exact 3D MTF $H(\mathbf{f})$ + noise-regularized frequency translation $T_{A \to B}(\mathbf{f}) = H_B(\mathbf{f}) H_A^*(\mathbf{f}) / (|H_A(\mathbf{f})|^2 + \lambda (1 + (\|\mathbf{f}\|/f_{\text{nyq}})^4))$ with strict DC conservation $T_{A \to B}(0) = 1$.
  - Exact $L^2 \to L^2$ operator spectral norm $\|T_{A \to B}\|_{2 \to 2} = \max_{\mathbf{f}} |T_{A \to B}(\mathbf{f})|$ gated by `TRANSLATION_NOISE_AMPLIFICATION_UNSAFE` (`> 2.0x`), plus blind NEMA RC-curve fingerprinting (`fingerprint_scanner_from_rc`).
- **EANM Gear-2018 Full-Covariance Budget & Fisher-Optimal STP SPECT/CT Designer (`stp_dosimetry_designer.py`)**:
  - Strictly positive-definite 7-parameter Gear et al. (2018) log-dose covariance matrix capturing camera calibration, dead-time/TEW scatter, CT-AC respiratory residual, PVE-RC sensitivity, volume–S-factor negative beta self-dose covariance cancellation ($\rho_{VS} = -0.96$), and TIAC uncertainty.
  - Exact population-prior Hänscheid (2018) & Madsen (2018) STP RMSE curves + late-timepoint Poisson counting decay + Cramér–Rao Fisher Information 2-timepoint schedule optimizer gated by `STP_PROTOCOL_INFEASIBLE_REQUIRE_MTP`.
