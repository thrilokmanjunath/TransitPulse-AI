# TransitPulse-AI — Data Feasibility & Provenance Report

## Executive Feasibility Decision
- **Geographic Scope:** Bengaluru Urban & Suburban Metropolitan Region, Karnataka, India
- **Transit Authority Under Investigation:** Bengaluru Metropolitan Transport Corporation (BMTC)
- **Feasibility Determination Date:** 2026-11-17
- **Primary Feasibility Determination:** **DUAL-MODE ARCHITECTURE MANDATED**
  - Static GTFS Route/Stop Topology: **Verified and Accessible**
  - Meteorological Data (Open-Meteo API): **Verified and Accessible**
  - Major Public Event Geometries: **Verified and Accessible**
  - Historical Continuous BMTC Real-Time Vehicle Position Feeds (GTFS-RT): **Unavailable / Restricted**
  - Operational Resolution: Deploy a calibrated, physics-grounded synthetic delay generator (`TransitDelaySimulator`) anchored to real BMTC routes and real Open-Meteo weather for rigorous, leak-free pipeline evaluation.

---

## 1. Comprehensive Audit of Target Data Sources

| Data Stream | Specific Source / Provider | Classification | Access Constraints & Details |
|---|---|---|---|
| **Transit Network Topology (GTFS Static)** | BMTC Open Transit Feeds / OpenCity.in / Bengaluru Mobility Repository | **Verified and Accessible** | Routes, stop coordinates, sequence orders, trip schedules, and route shapes are openly downloadable in GTFS CSV format without authentication. |
| **Transit Operational Telemetry (GTFS-RT)** | BMTC Intelligent Transport System (ITS) / Bangalore Smart City | **Unavailable / Restricted** | The BMTC ITS live streaming endpoint (`gtfs-rt`) is not publicly archived as open historical datasets. Live GPS feeds are restricted to internal applications (Namma BMTC app backend) and gated behind enterprise VPN / non-public authentication. No verified historical second-by-second vehicle telemetry is publicly available for long-term retrospective training. |
| **Historical Weather Observations** | Open-Meteo Historical Weather API (ERA5 Reanalysis) | **Verified and Accessible** | Open API, zero API key requirement, hourly resolution for Bengaluru coordinates (12.9716° N, 77.5946° E) covering precipitation, temperature, relative humidity, wind speed, gusts, and WMO weather codes. |
| **Operational Weather Forecasts** | Open-Meteo Forecast API (ECMWF & GFS Models) | **Verified and Accessible** | Open API providing 1-hour to 7-day forward predictions of precipitation and storm probabilities for leakage-free multi-horizon forecasting. |
| **Urban Event Geometry & Timetables** | Bengaluru Event Registry / Public Stadium Calendars (Chinnaswamy, Palace Grounds, BIEC, KTPO) | **Verified and Accessible** | Publicly accessible event schedules, coordinates, venue capacities, and start/end timestamps curated from verified public venue calendars. |
| **Road Network & Spatial Hierarchy** | OpenStreetMap (OSM) via Overpass / OSMnx / GeoPandas | **Verified and Accessible** | Open geodata under ODbL providing road segment classifications (primary, secondary, trunk), junction density, and spatial bus stop coordinates. |
| **Calibrated Delay Telemetry** | `TransitPulse-AI` Synthetic Physics Simulator | **Simulated Substitute** | Generates realistic, spatiotemporally autocorrelated delay distributions conditioned on real BMTC route topologies, real Open-Meteo precipitation shocks, and real stadium event crowds using BPR congestion curves. |

---

## 2. Scientific Honesty & Labeling Protocols

> [!IMPORTANT]
> **Strict Non-Misrepresentation Guarantee:**
> 1. Simulated delay observations are **never** presented or advertised as measured historical performance of the BMTC bus fleet.
> 2. All simulated datasets are flagged with `dataset_mode = "simulated"` in their Parquet metadata, database schemas, API responses, and dashboard headers.
> 3. Predictive evaluation metrics (MAE, RMSE, $R^2$) computed on simulated benchmarks are explicitly labeled as:
>    `Benchmark Type: Calibrated Simulation (Synthetic Ground Truth)`
> 4. The codebase provides plug-and-play schema-compatible connectors for real GTFS-Realtime feeds whenever credentials or enterprise API keys are supplied.

---

## 3. Calibrated Simulation Mathematical Framework

To ensure that the simulation mode provides a statistically defensible testbed for machine learning models and spatiotemporal feature interactions, delays are modeled as a multi-component stochastic process:

$$d_{s,t} = d_{s-1, t - \Delta t} \cdot \rho_{\text{decay}} + \delta_{\text{baseline}} + \delta_{\text{peak}} + \delta_{\text{weather}} + \delta_{\text{event}} + \epsilon_{s,t}$$

Where:
- $d_{s,t}$: Delay (in minutes) at stop sequence $s$ and time $t$.
- $d_{s-1, t - \Delta t}$: Delay inherited from the previous upstream stop on the route, decayed by transmission factor $\rho_{\text{decay}} \in [0.75, 0.95]$.
- $\delta_{\text{baseline}}$: Inherent operational route friction (log-normally distributed, $\mu = 1.2$, $\sigma = 0.4$).
- $\delta_{\text{peak}}$: Commuter rush-hour shock governed by a dual-Gaussian diurnal cycle peaking at 09:00 and 18:30 IST.
- $\delta_{\text{weather}}$: Meteorological friction penalty:
  $$\delta_{\text{weather}} = \alpha \cdot P_{t} + \beta \cdot \mathbf{1}_{\{\text{thunderstorm}\}}$$
  where $P_t$ is precipitation rate (mm/h) from Open-Meteo, with severe non-linear amplification when $P_t > 8.0$ mm/h.
- $\delta_{\text{event}}$: Spatial decay shockwave:
  $$\delta_{\text{event}} = \sum_{e \in \mathcal{E}} \frac{C_e}{1 + (r_{s,e} / r_0)^2} \cdot \exp\left(-\frac{(t - t_{e,\text{start}})^2}{2\sigma_e^2}\right)$$
  where $C_e$ represents venue capacity factor, $r_{s,e}$ is the geodesic distance from stop $s$ to venue $e$, $r_0 = 1000$ meters, and $t_{e,\text{start}}$ is the event start time.
- $\epsilon_{s,t} \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$: Uncorrelated random noise.

This formulation guarantees that the feature engineering pipeline, boosting algorithms, and SHAP explainability engines are subjected to authentic, non-linear spatiotemporal interactions without fabricating fake real-world agency performance.
