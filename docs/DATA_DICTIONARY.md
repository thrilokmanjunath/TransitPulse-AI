# TransitPulse-AI — Data Dictionary

This document provides schema specifications, semantic definitions, units of measurement, and availability timestamps for all raw, intermediate, and engineered feature columns in the TransitPulse-AI platform.

---

## 1. Raw Ingestion Schemas

### 1.1 Transit Static Entities (`data/raw/gtfs_*.parquet`)
| Column Name | Data Type | Nullable | Description & Domain |
|---|---|---|---|
| `route_id` | `string` | No | Unique identifier of transit route (e.g. `335-E`, `500-D`, `KIA-9`). |
| `route_short_name` | `string` | No | Public-facing route number or bus code. |
| `origin_name` | `string` | No | Route origin terminal (e.g. `Kempegowda Bus Station`). |
| `destination_name` | `string` | No | Route destination terminal (e.g. `Kadugodi Bus Stand`). |
| `stop_id` | `string` | No | Unique transit stop identifier. |
| `stop_name` | `string` | No | Name of bus stop. |
| `stop_sequence` | `int32` | No | Monotonically increasing 1-based stop sequence index along the route. |
| `stop_lat` | `float64` | No | WGS84 Latitude of bus stop (Domain: [12.70, 13.30]). |
| `stop_lon` | `float64` | No | WGS84 Longitude of bus stop (Domain: [77.30, 77.85]). |
| `scheduled_travel_time_sec` | `float64` | No | Scheduled traversal time in seconds from previous stop. |
| `cumulative_distance_km` | `float64` | No | Cumulative distance from route origin along road path in kilometers. |

### 1.2 Meteorological Observations (`data/raw/weather_*.parquet`)
| Column Name | Data Type | Unit | Description |
|---|---|---|---|
| `timestamp` | `datetime64[ns, Asia/Kolkata]` | — | ISO-8601 observation timestamp in IST. |
| `temperature_2m` | `float32` | °C | Dry bulb air temperature at 2 meters above ground. |
| `relative_humidity_2m` | `float32` | % | Relative humidity percentage (Domain: [0, 100]). |
| `precipitation` | `float32` | mm | Total liquid water precipitation equivalent over the preceding hour. |
| `rain` | `float32` | mm | Liquid rain depth over the preceding hour. |
| `wind_speed_10m` | `float32` | km/h | Sustained wind speed at 10 meters above ground. |
| `wind_gusts_10m` | `float32` | km/h | Maximum wind gust speed over preceding hour. |
| `weather_code` | `int16` | WMO Code | World Meteorological Organization synoptic weather code. |

### 1.3 Urban Event Entities (`data/raw/events_*.parquet`)
| Column Name | Data Type | Nullable | Description |
|---|---|---|---|
| `event_id` | `string` | No | Unique identifier for the urban event. |
| `event_name` | `string` | No | Title of event (e.g., `IPL Match: RCB vs CSK`). |
| `venue_name` | `string` | No | Name of venue (e.g., `M. Chinnaswamy Stadium`). |
| `event_category` | `string` | No | Category (`sports`, `concert`, `exhibition`, `festival`, `summit`). |
| `venue_lat` | `float64` | No | Latitude of event venue. |
| `venue_lon` | `float64` | No | Longitude of event venue. |
| `start_time` | `datetime64[ns, Asia/Kolkata]` | No | Scheduled kickoff or gates-open timestamp. |
| `end_time` | `datetime64[ns, Asia/Kolkata]` | No | Scheduled conclusion timestamp. |
| `expected_attendance` | `int32` | Yes | Verified venue capacity or ticketed attendance count. |

---

## 2. Engineered Spatiotemporal Feature Matrix (`data/processed/features_*.parquet`)

| Feature Name | Type | Source Domain | Temporal Availability Cutoff | Leakage Risk & Mitigation |
|---|---|---|---|---|
| `hour_of_day` | `int32` | Temporal | $T_{\text{pred}}$ | Zero risk: deterministic timestamp derivation. |
| `day_of_week` | `int32` | Temporal | $T_{\text{pred}}$ | Zero risk: deterministic. |
| `is_weekend` | `int8` | Temporal | $T_{\text{pred}}$ | Zero risk: deterministic. |
| `is_peak_hour` | `int8` | Temporal | $T_{\text{pred}}$ | Zero risk: morning (08:00–11:00) and evening (17:00–21:00) IST. |
| `hour_sin`, `hour_cos` | `float32` | Temporal | $T_{\text{pred}}$ | Zero risk: cyclical transformation of `hour_of_day`. |
| `precip_rate_mm` | `float32` | Weather | $T_{\text{pred}} - 2\text{m}$ | Forecast vintage used for forward horizons; reanalysis for past. |
| `precip_rolling_3h_mm` | `float32` | Weather | $T_{\text{pred}} - 2\text{m}$ | Rolling sum strictly backwards over $[T_{\text{pred}}-3\text{h}, T_{\text{pred}}]$. |
| `wind_gust_kmh` | `float32` | Weather | $T_{\text{pred}} - 2\text{m}$ | Strictly backwards observation. |
| `weather_severity_idx` | `int8` | Weather | $T_{\text{pred}} - 2\text{m}$ | Ordinal mapping from WMO codes (0=clear to 3=severe thunderstorm). |
| `rain_peak_interaction` | `float32` | Interaction | $T_{\text{pred}} - 2\text{m}$ | `precip_rate_mm * is_peak_hour`. Non-linear congestion multiplier. |
| `event_count_500m` | `int32` | Spatial/Event | $T_{\text{pred}}$ | Pre-scheduled public calendar entries within 500m buffer. |
| `event_count_1km` | `int32` | Spatial/Event | $T_{\text{pred}}$ | Pre-scheduled public calendar entries within 1km buffer. |
| `event_count_3km` | `int32` | Spatial/Event | $T_{\text{pred}}$ | Pre-scheduled public calendar entries within 3km buffer. |
| `dist_nearest_event_km` | `float32` | Spatial/Event | $T_{\text{pred}}$ | Geodesic distance (km) to nearest active/upcoming venue. |
| `min_to_event_start` | `float32` | Spatial/Event | $T_{\text{pred}}$ | Minutes between $T_{\text{pred}}$ and nearest venue start time. |
| `scheduled_travel_time_sec` | `float32` | Transit | Static | Pre-published timetable schedule. |
| `stop_sequence` | `int32` | Transit | Static | Position along route path. |
| `prev_stop_delay_min` | `float32` | Transit Lag | $T_{\text{pred}} - 2\text{m}$ | Observed arrival delay at immediately preceding stop $s-1$. Missing on origin stop, imputed with route historical mean. |
| `route_hist_mean_delay_min` | `float32` | Transit Lag | $T_{\text{pred}} - 2\text{m}$ | Out-of-fold historical average delay for route at matching time-of-day. |
| `stop_density_1km` | `float32` | Spatial | Static | Count of transit stops within 1 km radius. |
| `road_network_hierarchy` | `int8` | Spatial | Static | Road classification (1=tertiary, 2=secondary, 3=primary/arterial, 4=trunk/highway). |

---

## 3. Target Variables

| Target Column | Type | Unit / Classes | Description |
|---|---|---|---|
| `delay_minutes` | `float32` | Minutes | Continuous delay (positive = late, negative = early). |
| `delay_category` | `string` | 4 classes | Multi-class label: `On-time` ($\le 5\text{m}$), `Minor` ($5\text{m} < d \le 10\text{m}$), `Moderate` ($10\text{m} < d \le 20\text{m}$), `Severe` ($> 20\text{m}$). |
| `target_horizon` | `int32` | Minutes | Prediction horizon in minutes ($H \in \{15, 30, 60\}$). |
| `dataset_mode` | `string` | Categorical | `observed` or `simulated`. |
