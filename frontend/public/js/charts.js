/**
 * charts.js
 * Reads all data from <script id="chart-data" type="application/json">
 * so there are zero HTML-attribute encoding issues.
 * Served from /js/charts.js (local public folder, no CDN).
 */

(function () {
    'use strict';

    /* ── Read embedded JSON data ──────────────────────────── */
    const dataEl = document.getElementById('chart-data');
    if (!dataEl) return;

    let chartData;
    try { chartData = JSON.parse(dataEl.textContent); }
    catch (e) { console.error('Inspector Pro: failed to parse chart-data JSON', e); return; }

    const { riskPct = 0, features = [], inputs = {} } = chartData;

    /* ── Colour palette ───────────────────────────────────── */
    const AMBER       = '#f7ba2b';
    const CORAL       = '#ea5358';
    const GOOD        = '#22c55e';
    const DANGER      = '#ef4444';
    const WARN        = '#f59e0b';
    const INFO        = '#38bdf8';
    const SURFACE_LN  = 'rgba(255,255,255,0.07)';

    function featureColor(direction) {
        if (direction === 'negative') return DANGER;
        if (direction === 'positive') return GOOD;
        return WARN;
    }

    /* ── Apply global Chart.js defaults ──────────────────── */
    Chart.defaults.font.family  = "'Segoe UI', system-ui, sans-serif";
    Chart.defaults.font.size    = 12;
    Chart.defaults.color        = '#a1a1aa';
    Chart.defaults.borderColor  = SURFACE_LN;
    Chart.defaults.responsive   = true;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.animation.duration  = 900;

    const tooltipDefaults = {
        backgroundColor: 'rgba(10,10,14,0.97)',
        borderColor:      AMBER,
        borderWidth:      1,
        cornerRadius:     8,
        padding:          10,
        titleColor:       '#f4f4f5',
        bodyColor:        '#a1a1aa',
        boxPadding:       4,
    };

    /* ── 1. Risk Gauge  ───────────────────────────────────── */
    function buildGauge() {
        const el = document.getElementById('risk-gauge-chart');
        if (!el) return;

        const risk = parseFloat(riskPct);
        const safe = 100 - risk;

        /* color ramp: green → amber → red */
        let riskColor = GOOD;
        if (risk >= 60) riskColor = DANGER;
        else if (risk >= 30) riskColor = WARN;

        new Chart(el, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [risk, safe],
                    backgroundColor: [riskColor, 'rgba(255,255,255,0.06)'],
                    borderWidth:     0,
                    hoverOffset:     4,
                    circumference:   240,
                    rotation:        240,
                }]
            },
            options: {
                cutout: '78%',
                plugins: {
                    legend:  { display: false },
                    tooltip: {
                        ...tooltipDefaults,
                        callbacks: {
                            label: ctx => ctx.dataIndex === 0
                                ? ` Failure Risk: ${risk.toFixed(1)}%`
                                : ` Safe Zone: ${safe.toFixed(1)}%`,
                        }
                    }
                }
            }
        });
    }

    /* ── 2. Feature Impact — horizontal bar ───────────────── */
    function buildFeatureBar() {
        const el = document.getElementById('feature-impact-chart');
        if (!el || !features.length) return;

        /* cap at top 7 for readability */
        const slice   = features.slice(0, 7);
        const labels  = slice.map(f => f.name);
        const values  = slice.map(f => parseFloat(f.impact));
        const colors  = slice.map(f => featureColor(f.direction));

        /* normalise to max = 100 for bar display (raw impact can exceed 100%) */
        const maxVal  = Math.max(...values, 1);
        const normVals = values.map(v => (v / maxVal) * 100);

        new Chart(el, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label:           'Influence',
                    data:            normVals,
                    backgroundColor: colors.map(c => c + 'bb'),
                    borderColor:     colors,
                    borderWidth:     1,
                    borderRadius:    6,
                    borderSkipped:   false,
                }]
            },
            options: {
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        ...tooltipDefaults,
                        callbacks: {
                            label: (ctx) => {
                                const f = slice[ctx.dataIndex];
                                const sign = f.raw_shap >= 0 ? '+' : '';
                                return [
                                    ` SHAP: ${sign}${f.raw_shap.toFixed(4)}`,
                                    ` Influence: ${f.impact.toFixed(1)}%`,
                                ];
                            },
                            title: ctx => ctx[0].label,
                        }
                    }
                },
                scales: {
                    x: {
                        grid:  { color: SURFACE_LN },
                        ticks: { display: false },
                        border:{ display: false },
                        max:   100,
                    },
                    y: {
                        grid:  { color: 'transparent' },
                        ticks: { color: '#c4c4cc', font: { weight: '600', size: 11 } },
                        border:{ display: false },
                    }
                }
            }
        });
    }

    /* ── 3. Risk Breakdown — polar area ───────────────────── */
    function buildPolarArea() {
        const el = document.getElementById('risk-breakdown-chart');
        if (!el || !features.length) return;

        const slice  = features.slice(0, 6);
        const labels = slice.map(f => f.name);
        const values = slice.map(f => parseFloat(f.impact));
        const colors = slice.map(f => featureColor(f.direction) + 'bb');
        const bords  = slice.map(f => featureColor(f.direction));

        new Chart(el, {
            type: 'polarArea',
            data: {
                labels,
                datasets: [{
                    data:            values,
                    backgroundColor: colors,
                    borderColor:     bords,
                    borderWidth:     1,
                }]
            },
            options: {
                plugins: {
                    legend: {
                        display:  true,
                        position: 'bottom',
                        labels: {
                            color:    '#a1a1aa',
                            font:     { size: 10 },
                            padding:  10,
                            boxWidth: 10,
                            boxHeight:10,
                        }
                    },
                    tooltip: {
                        ...tooltipDefaults,
                        callbacks: {
                            label: ctx => ` ${ctx.parsed.r.toFixed(1)}% weight in decision`,
                        }
                    }
                },
                scales: {
                    r: {
                        ticks: { display: false },
                        grid:  { color: SURFACE_LN }
                    }
                }
            }
        });
    }

    /* ── 4. Sensor Radar ──────────────────────────────────── */
    function buildRadar() {
        const el = document.getElementById('sensor-radar-chart');
        if (!el) return;

        const sensorKeys = Object.keys(inputs);
        if (!sensorKeys.length) return;

        /* safe operating ranges for normalisation */
        const ranges = {
            Temperature:       [20,  120],
            Pressure:          [0,   150],
            Speed:             [0,  3000],
            Vibration:         [0,   10],
            Humidity:          [0,  100],
            Power_Consumption: [0,  300],
            Material_Hardness: [0,  100],
        };
        const axisLabels = {
            Temperature:       'Temp (°C)',
            Pressure:          'Pressure',
            Speed:             'Speed',
            Vibration:         'Vibration',
            Humidity:          'Humidity',
            Power_Consumption: 'Power',
            Material_Hardness: 'Hardness',
        };

        const labels = sensorKeys.map(k => axisLabels[k] || k);
        const values = sensorKeys.map(k => {
            const [lo, hi] = ranges[k] || [0, 100];
            return Math.min(100, Math.max(0, ((inputs[k] - lo) / (hi - lo)) * 100));
        });
        const rawValues = sensorKeys.map(k => inputs[k]);

        new Chart(el, {
            type: 'radar',
            data: {
                labels,
                datasets: [{
                    label:              'Sensor Reading',
                    data:               values,
                    backgroundColor:    'rgba(247,186,43,0.10)',
                    borderColor:        AMBER,
                    borderWidth:        2,
                    pointBackgroundColor: AMBER,
                    pointRadius:        4,
                    pointHoverRadius:   7,
                }]
            },
            options: {
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks:       { display: false, stepSize: 25 },
                        grid:        { color: SURFACE_LN },
                        angleLines:  { color: SURFACE_LN },
                        pointLabels: {
                            color: '#c4c4cc',
                            font:  { size: 12, weight: '600' },
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        ...tooltipDefaults,
                        callbacks: {
                            title: ctx => labels[ctx[0].dataIndex],
                            label: ctx => {
                                const k = sensorKeys[ctx.dataIndex];
                                const u = { Temperature:'°C', Pressure:' PSI', Speed:' RPM',
                                            Vibration:' mm/s', Humidity:'%',
                                            Power_Consumption:' kW', Material_Hardness:'' };
                                return ` ${rawValues[ctx.dataIndex]}${u[k] || ''}   (${ctx.parsed.r.toFixed(0)}% of range)`;
                            }
                        }
                    }
                }
            }
        });
    }

    /* ── 5. Animate SHAP bar fills ───────────────────────── */
    function animateBars() {
        document.querySelectorAll('[data-fill-width]').forEach(bar => {
            const target = bar.getAttribute('data-fill-width');
            bar.style.width = '0%';
            /* short delay so the CSS transition has something to animate from */
            setTimeout(() => { bar.style.width = target; }, 120);
        });
    }

    /* ── Boot ─────────────────────────────────────────────── */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init(); /* DOM already ready (scripts at end of body) */
    }

    function init() {
        buildGauge();
        buildFeatureBar();
        buildPolarArea();
        buildRadar();
        animateBars();
    }

}());
