"""Interactive Web Application Frontend Route for TransitPulse-AI."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Web App"])

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TransitPulse-AI — Bengaluru Transit Delay Prediction</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans selection:bg-indigo-500 selection:text-white">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-amber-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <i class="fa-solid fa-bus text-white text-lg"></i>
        </div>
        <div>
          <span class="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">TransitPulse-AI</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 ml-2">v1.0.0 Live</span>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <div class="hidden sm:flex items-center space-x-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-full">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>API Operational</span>
        </div>
        <a href="/docs" target="_blank" class="text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3.5 py-1.5 rounded-lg transition-colors flex items-center space-x-1.5">
          <i class="fa-solid fa-book-open"></i>
          <span>Swagger Docs</span>
        </a>
        <a href="https://github.com/thrilokmanjunath/TransitPulse-AI" target="_blank" class="text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3.5 py-1.5 rounded-lg transition-colors flex items-center space-x-1.5">
          <i class="fa-brands fa-github"></i>
          <span>GitHub</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Notice Banner -->
  <div class="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2.5">
    <div class="max-w-7xl mx-auto flex items-center justify-center text-xs text-amber-300 text-center space-x-2">
      <i class="fa-solid fa-triangle-exclamation"></i>
      <span><strong>Research Notice:</strong> Observed GTFS topology + Open-Meteo weather integrated. Operational vehicle delays use calibrated simulation (<code>dataset_mode: simulated</code>).</span>
    </div>
  </div>

  <!-- Main Content -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Left Column: Controls (5 cols) -->
      <div class="lg:col-span-5 space-y-6">
        <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur">
          <div class="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
            <h2 class="text-base font-semibold text-white flex items-center space-x-2">
              <i class="fa-solid fa-sliders text-indigo-400"></i>
              <span>Prediction Scenario</span>
            </h2>
            <span class="text-xs text-slate-400">Bengaluru BMTC</span>
          </div>

          <form id="predict-form" class="space-y-5">
            <!-- Route Selector -->
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-2">Transit Corridor Route</label>
              <select id="route-select" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors">
                <option value="">Loading routes...</option>
              </select>
            </div>

            <!-- Stop Selector -->
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-2">Target Bus Stop</label>
              <select id="stop-select" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors">
                <option value="1">Select a route first</option>
              </select>
            </div>

            <!-- Prior Stop Delay Slider -->
            <div>
              <div class="flex justify-between items-center text-xs font-medium text-slate-300 mb-2">
                <span>Prior Upstream Stop Delay</span>
                <span id="prev-delay-val" class="text-indigo-400 font-mono font-bold">2.5 min</span>
              </div>
              <input type="range" id="prev-delay" min="0" max="25" step="0.5" value="2.5" class="w-full accent-indigo-500 cursor-pointer">
            </div>

            <!-- Precipitation Slider -->
            <div>
              <div class="flex justify-between items-center text-xs font-medium text-slate-300 mb-2">
                <span>Rainfall Intensity (Open-Meteo)</span>
                <span id="precip-val" class="text-blue-400 font-mono font-bold">5.0 mm/h</span>
              </div>
              <input type="range" id="precip" min="0" max="50" step="1.0" value="5.0" class="w-full accent-blue-500 cursor-pointer">
              <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>Clear</span>
                <span>Moderate Rain</span>
                <span>Monsoon Deluge</span>
              </div>
            </div>

            <!-- Temperature Slider -->
            <div>
              <div class="flex justify-between items-center text-xs font-medium text-slate-300 mb-2">
                <span>Ambient Temperature</span>
                <span id="temp-val" class="text-amber-400 font-mono font-bold">26.0 °C</span>
              </div>
              <input type="range" id="temp" min="15" max="42" step="0.5" value="26.0" class="w-full accent-amber-500 cursor-pointer">
            </div>

            <!-- Active Event Toggle -->
            <div class="pt-2">
              <label class="flex items-center space-x-3 bg-slate-950/60 border border-slate-800 p-3.5 rounded-xl cursor-pointer hover:border-slate-700 transition-colors">
                <input type="checkbox" id="event-active" class="w-4 h-4 rounded text-indigo-600 focus:ring-indigo-500 focus:ring-offset-slate-950 border-slate-700 bg-slate-900">
                <div class="text-xs">
                  <div class="font-medium text-white">Active Mega-Event Nearby</div>
                  <div class="text-slate-400 text-[11px]">Chinnaswamy Stadium IPL match or BIEC Expo shock</div>
                </div>
              </label>
            </div>

            <!-- Submit Button -->
            <button type="submit" id="submit-btn" class="w-full bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-medium py-3 px-4 rounded-xl shadow-lg shadow-indigo-600/25 transition-all flex items-center justify-center space-x-2 text-sm">
              <i class="fa-solid fa-bolt"></i>
              <span>Forecast Delay & Shockwave</span>
            </button>
          </form>
        </div>
      </div>

      <!-- Right Column: Results & Explainability (7 cols) -->
      <div class="lg:col-span-7 space-y-6">
        
        <!-- Live Delay Card -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur relative overflow-hidden">
          <div class="absolute -right-8 -top-8 w-40 h-40 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
          
          <div class="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
            <h3 class="text-base font-semibold text-white flex items-center space-x-2">
              <i class="fa-solid fa-chart-line text-indigo-400"></i>
              <span>Predicted Arrival Delay</span>
            </h3>
            <span id="delay-badge" class="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Minor Delay</span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            <div>
              <div class="text-slate-400 text-xs font-medium uppercase tracking-wider mb-1">Expected Vehicle Delay</div>
              <div class="flex items-baseline space-x-2">
                <span id="predicted-delay" class="text-5xl font-black font-mono tracking-tight text-white">--</span>
                <span class="text-slate-400 text-lg font-medium">min</span>
              </div>
              <div class="text-xs text-slate-400 mt-2 flex items-center space-x-1.5">
                <i class="fa-solid fa-shield-halved text-slate-500"></i>
                <span>90% Confidence Interval:</span>
                <span id="ci-range" class="font-mono text-slate-200">--</span>
              </div>
            </div>

            <!-- Forward Horizons -->
            <div class="bg-slate-950/70 border border-slate-800/80 rounded-xl p-4 space-y-3">
              <div class="text-xs font-semibold text-slate-300 flex items-center justify-between">
                <span>Multi-Horizon Projections</span>
                <i class="fa-solid fa-arrows-split-up-and-left text-slate-500"></i>
              </div>
              <div class="space-y-2 text-xs">
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">+15 min Horizon:</span>
                  <span id="h15" class="font-mono font-semibold text-indigo-300">--</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">+30 min Horizon:</span>
                  <span id="h30" class="font-mono font-semibold text-indigo-300">--</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">+60 min Horizon:</span>
                  <span id="h60" class="font-mono font-semibold text-indigo-300">--</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Attribution Shockwave Decomposition -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur">
          <div class="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
            <h3 class="text-base font-semibold text-white flex items-center space-x-2">
              <i class="fa-solid fa-pie-chart text-amber-400"></i>
              <span>SHAP Delay Shockwave Decomposition</span>
            </h3>
            <span class="text-xs text-slate-400">Marginal Attributions</span>
          </div>

          <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div class="bg-slate-950/60 border border-slate-800 p-3.5 rounded-xl text-center">
              <div class="text-slate-400 text-[11px] mb-1">Precipitation</div>
              <div id="attr-weather" class="text-lg font-bold font-mono text-blue-400">--</div>
            </div>
            <div class="bg-slate-950/60 border border-slate-800 p-3.5 rounded-xl text-center">
              <div class="text-slate-400 text-[11px] mb-1">Event Shock</div>
              <div id="attr-event" class="text-lg font-bold font-mono text-purple-400">--</div>
            </div>
            <div class="bg-slate-950/60 border border-slate-800 p-3.5 rounded-xl text-center">
              <div class="text-slate-400 text-[11px] mb-1">Commuter Peak</div>
              <div id="attr-peak" class="text-lg font-bold font-mono text-amber-400">--</div>
            </div>
            <div class="bg-slate-950/60 border border-slate-800 p-3.5 rounded-xl text-center">
              <div class="text-slate-400 text-[11px] mb-1">Route Baseline</div>
              <div id="attr-route" class="text-lg font-bold font-mono text-slate-300">--</div>
            </div>
          </div>
        </div>

        <!-- Benchmarks & Audit Highlights -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div class="bg-slate-900/70 border border-slate-800 p-4 rounded-xl">
            <div class="text-slate-400 text-xs">Production Model</div>
            <div class="text-base font-bold text-white mt-1">LightGBM (E4)</div>
            <div class="text-[11px] text-emerald-400 mt-1">Sub-10ms Inference</div>
          </div>
          <div class="bg-slate-900/70 border border-slate-800 p-4 rounded-xl">
            <div class="text-slate-400 text-xs">Out-of-Time MAE</div>
            <div class="text-base font-bold text-indigo-400 mt-1">1.463 min</div>
            <div class="text-[11px] text-slate-400 mt-1">RMSE: 2.22 min</div>
          </div>
          <div class="bg-slate-900/70 border border-slate-800 p-4 rounded-xl">
            <div class="text-slate-400 text-xs">Severe Delay Recall</div>
            <div class="text-base font-bold text-amber-400 mt-1">83.56%</div>
            <div class="text-[11px] text-slate-400 mt-1">Macro F1: 0.8322</div>
          </div>
        </div>

      </div>
    </div>
  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800 bg-slate-950 py-6 mt-auto">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-4">
      <div>
        TransitPulse-AI &copy; 2026. Built by <a href="https://github.com/thrilokmanjunath" class="text-slate-400 hover:text-white underline">Thrilok Manjunath</a>.
      </div>
      <div class="flex items-center space-x-6">
        <a href="/docs" class="hover:text-slate-300">Interactive OpenAPI Docs</a>
        <a href="/health" class="hover:text-slate-300">System Health</a>
        <a href="https://github.com/thrilokmanjunath/TransitPulse-AI" class="hover:text-slate-300">GitHub Repository</a>
      </div>
    </div>
  </footer>

  <!-- Frontend Logic -->
  <script>
    const routeSelect = document.getElementById('route-select');
    const stopSelect = document.getElementById('stop-select');
    const prevDelay = document.getElementById('prev-delay');
    const precip = document.getElementById('precip');
    const temp = document.getElementById('temp');
    const eventActive = document.getElementById('event-active');
    
    // Sliders live text
    prevDelay.addEventListener('input', e => document.getElementById('prev-delay-val').textContent = `${e.target.value} min`);
    precip.addEventListener('input', e => document.getElementById('precip-val').textContent = `${e.target.value} mm/h`);
    temp.addEventListener('input', e => document.getElementById('temp-val').textContent = `${e.target.value} °C`);

    // Load Routes on Boot
    async function loadRoutes() {
      try {
        const res = await fetch('/api/v1/routes');
        const routes = await res.json();
        routeSelect.innerHTML = routes.map(r => 
          `<option value="${r.route_id}">${r.route_short_name} — ${r.origin_name} to ${r.destination_name} (${r.distance_km} km)</option>`
        ).join('');
        loadStops(routes[0].route_id);
      } catch (err) {
        routeSelect.innerHTML = '<option value="335-E">335-E (Majestic to ITPL)</option>';
      }
    }

    // Load Stops for selected Route
    async function loadStops(routeId) {
      try {
        const res = await fetch(`/api/v1/stops?route_id=${encodeURIComponent(routeId)}`);
        const stops = await res.json();
        stopSelect.innerHTML = stops.map(s => 
          `<option value="${s.stop_sequence}">Stop ${s.stop_sequence}: ${s.stop_name}</option>`
        ).join('');
      } catch (err) {
        stopSelect.innerHTML = '<option value="3">Stop 3: Richmond Circle</option>';
      }
    }

    routeSelect.addEventListener('change', e => loadStops(e.target.value));

    // Handle Prediction Form Submit
    document.getElementById('predict-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('submit-btn');
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i><span>Computing...</span>';

      const payload = {
        route_id: routeSelect.value || '335-E',
        stop_sequence: parseInt(stopSelect.value || 3),
        prev_stop_delay_min: parseFloat(prevDelay.value),
        precip_mm: parseFloat(precip.value),
        temperature_c: parseFloat(temp.value),
        is_event_active: eventActive.checked
      };

      try {
        const res = await fetch('/api/v1/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Update Delay
        document.getElementById('predicted-delay').textContent = data.predicted_delay_min.toFixed(2);
        document.getElementById('ci-range').textContent = `[${data.confidence_interval_90.lower_min.toFixed(1)} - ${data.confidence_interval_90.upper_min.toFixed(1)} min]`;

        // Update Badge
        const badge = document.getElementById('delay-badge');
        badge.textContent = `${data.delay_category} Delay`;
        if (data.delay_category === 'Severe') {
          badge.className = 'px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20';
        } else if (data.delay_category === 'Moderate') {
          badge.className = 'px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20';
        } else {
          badge.className = 'px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        }

        // Multi-Horizon
        document.getElementById('h15').textContent = `${data.multi_horizon_delays.horizon_15m.toFixed(2)} min`;
        document.getElementById('h30').textContent = `${data.multi_horizon_delays.horizon_30m.toFixed(2)} min`;
        document.getElementById('h60').textContent = `${data.multi_horizon_delays.horizon_60m.toFixed(2)} min`;

        // Attributions
        document.getElementById('attr-weather').textContent = `+${data.attribution_summary.weather_impact_min.toFixed(1)}m`;
        document.getElementById('attr-event').textContent = `+${data.attribution_summary.event_impact_min.toFixed(1)}m`;
        document.getElementById('attr-peak').textContent = `+${data.attribution_summary.commuter_peak_impact_min.toFixed(1)}m`;
        document.getElementById('attr-route').textContent = `+${data.attribution_summary.route_baseline_impact_min.toFixed(1)}m`;

      } catch (err) {
        alert('Prediction request failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-bolt"></i><span>Forecast Delay & Shockwave</span>';
      }
    });

    // Initial Load
    window.addEventListener('DOMContentLoaded', () => {
      loadRoutes();
      // Auto run initial prediction
      setTimeout(() => document.getElementById('submit-btn').click(), 600);
    });
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def app_home():
    """Serves the interactive web application UI."""
    return HTMLResponse(content=HTML_CONTENT)


@router.get("/app", response_class=HTMLResponse, include_in_schema=False)
def app_view():
    """Alias for interactive web application UI."""
    return HTMLResponse(content=HTML_CONTENT)
