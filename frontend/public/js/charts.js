/**
 * charts.js
 * Reads all the AI output data safely passed from our backend via a hidden JSON script block.
 * This prevents messy HTML-attribute encoding issues.
 * Served from /js/charts.js (local public folder, no CDN).
 *
 * Charts:
 *   1. Risk Gauge (doughnut)
 *   2. Feature Impact (horizontal bar)
 *   3. Risk Share (polar area)
 *   4. SHAP Waterfall (Power BI style stacked horizontal bar)
 *   5. Sensor Radar
 *   6. SHAP bar animations
 *   7. Interactive sortable table + CSV export
 */

(function () {
    'use strict';

    /* ── Read embedded JSON data ──────────────────────────── */
    const dataEl = document.getElementById('chart-data');
    if (!dataEl) return;

    let chartData;
    try { chartData = JSON.parse(dataEl.textContent); }
    catch (e) { console.error('Inspector Pro: failed to parse chart-data JSON', e); return; }

    const {
        riskPct          = 0,
        features         = [],
        inputs           = {},
        modelConfidence  = 0,
        topRiskFactor    = 'N/A',
        riskFactorsCount = 0,
        isFail           = false,
    } = chartData;

    /* ── Colour palette ───────────────────────────────────── */
    const AMBER       = '#f7ba2b';
    const CORAL       = '#ea5358';
    const GOOD        = '#22c55e';
    const DANGER      = '#ef4444';
    const WARN        = '#f59e0b';
    const INFO        = '#38bdf8';
    const SURFACE_LN  = 'rgba(255,255,255,0.07)';
    const PURPLE      = '#a78bfa';

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

        /* Establish conditional coloring dynamically: green for safe, amber for warning, and red for danger */
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

        /* Isolate the top 7 factors influencing the decision to keep the bar chart visually clean and easy to read */
        const slice   = features.slice(0, 7);
        const labels  = slice.map(f => f.name);
        const values  = slice.map(f => parseFloat(f.impact));
        const colors  = slice.map(f => featureColor(f.direction));

        /* Normalize values relative to the maximum impact score so the bars scale cleanly within the 100% width container */
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

    /* ── 4. SHAP Waterfall — Power BI style ───────────────── */
    function buildWaterfall() {
        const el = document.getElementById('waterfall-chart');
        if (!el || !features.length) return;

        /* Order the SHAP values by their absolute magnitude to highlight only the 7 most impactful variables */
        const sorted = [...features]
            .sort((a, b) => Math.abs(b.raw_shap) - Math.abs(a.raw_shap))
            .slice(0, 7);

        /* Construct a cumulative waterfall effect where each feature's bar builds upon the endpoint of the preceding feature */
        const labels    = [];
        const starts    = [];  // floating bar start
        const barValues = [];  // floating bar length (positive = up, negative = down)
        const barColors = [];

        let runningBase = 0;

        sorted.forEach(f => {
            const contribution = f.raw_shap; // raw SHAP contribution
            labels.push(f.name);
            starts.push(runningBase);
            barValues.push(contribution);
            barColors.push(featureColor(f.direction) + 'cc');
            runningBase += contribution;
        });

        /* Add a "Total Risk" bar anchored at 0 */
        labels.push('Total Risk');
        starts.push(0);
        barValues.push(runningBase);
        barColors.push(isFail ? DANGER + 'cc' : GOOD + 'cc');

        /* We create the floating waterfall illusion by layering a transparent 'offset' bar underneath the visible colored bar.
           This allows us to mimic true Power BI waterfall charts using standard Chart.js stacked bars.
           The offset is transparent and pushes the visible colored dataset to its correct starting place. */
        const offsetData = starts.map((s, i) => {
            const v = barValues[i];
            return v >= 0 ? s : s + v;  // for negatives, anchor at lower edge
        });
        const absValues = barValues.map(v => Math.abs(v));

        new Chart(el, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label:           'Offset',
                        data:            offsetData,
                        backgroundColor: 'transparent',
                        borderWidth:     0,
                        borderSkipped:   false,
                    },
                    {
                        label:           'SHAP Contribution',
                        data:            absValues,
                        backgroundColor: barColors,
                        borderColor:     barColors.map(c => c.replace('cc','ff')),
                        borderWidth:     1,
                        borderRadius:    5,
                        borderSkipped:   false,
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        ...tooltipDefaults,
                        callbacks: {
                            title: ctx => ctx[0].label,
                            label: ctx => {
                                if (ctx.datasetIndex === 0) return null; // hide offset
                                const idx = ctx.dataIndex;
                                if (idx === sorted.length) {
                                    return ` Total cumulative risk: ${runningBase.toFixed(4)}`;
                                }
                                const f = sorted[idx];
                                const sign = f.raw_shap >= 0 ? '+' : '';
                                return [
                                    ` SHAP: ${sign}${f.raw_shap.toFixed(4)}`,
                                    ` Direction: ${f.direction === 'negative' ? '▲ Raises Risk' : '▼ Lowers Risk'}`,
                                ];
                            }
                        }
                    },
                    title: {
                        display: false,
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        grid:    { color: SURFACE_LN },
                        ticks:   { color: '#a1a1aa', font: { size: 10 } },
                        border:  { display: false },
                    },
                    y: {
                        stacked: true,
                        grid:    { color: 'transparent' },
                        ticks:   { color: '#c4c4cc', font: { weight: '600', size: 11 } },
                        border:  { display: false },
                    }
                }
            }
        });
    }

    /* ── 5. Sensor Radar ──────────────────────────────────── */
    function buildRadar() {
        const el = document.getElementById('sensor-radar-chart');
        if (!el) return;

        const sensorKeys = Object.keys(inputs);
        if (!sensorKeys.length) return;

        /* Define the standard physical operating bounds for each sensor to normalize them onto the 0-100% radar scale */
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

    /* ── 6. Animate SHAP bar fills ───────────────────────── */
    function animateBars() {
        document.querySelectorAll('[data-fill-width]').forEach(bar => {
            const target = bar.getAttribute('data-fill-width');
            bar.style.width = '0%';
            setTimeout(() => { bar.style.width = target; }, 120);
        });
    }

    /* ── 7a. Sortable Table ───────────────────────────────── */
    function initSortableTable() {
        const table = document.getElementById('shap-table');
        if (!table) return;

        const tbody = table.querySelector('tbody');
        const headers = table.querySelectorAll('th.sortable');
        let sortCol = 3;   // default: sort by Influence %
        let sortAsc = false;

        headers.forEach(th => {
            th.addEventListener('click', () => {
                const col  = parseInt(th.dataset.col, 10);
                const type = th.dataset.type;

                if (sortCol === col) {
                    sortAsc = !sortAsc;
                } else {
                    sortCol = col;
                    sortAsc = true;
                }

                /* Refresh the up/down directional arrows on all table headers to reflect the newly active sort column */
                headers.forEach(h => {
                    const icon = h.querySelector('.sort-icon');
                    if (!icon) return;
                    if (parseInt(h.dataset.col, 10) === col) {
                        icon.textContent = sortAsc ? '↑' : '↓';
                        h.classList.add('sorted');
                    } else {
                        icon.textContent = '⇅';
                        h.classList.remove('sorted');
                    }
                });

                sortRows(tbody, col, type, sortAsc);
            });
        });

        /* Automatically organize the table by 'Influence %' as soon as the page finishes loading */
        const defaultIcon = table.querySelector(`th[data-col="${sortCol}"] .sort-icon`);
        if (defaultIcon) defaultIcon.textContent = '↓';
        sortRows(tbody, sortCol, 'number', sortAsc);
    }

    function sortRows(tbody, colIdx, type, asc) {
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
            const cells = 'td';
            const aCell = a.querySelectorAll('td')[colIdx];
            const bCell = b.querySelectorAll('td')[colIdx];
            if (!aCell || !bCell) return 0;

            let aVal = aCell.textContent.trim();
            let bVal = bCell.textContent.trim();

            if (type === 'number') {
                aVal = parseFloat(aVal.replace(/[^0-9.\-]/g, '')) || 0;
                bVal = parseFloat(bVal.replace(/[^0-9.\-]/g, '')) || 0;
                return asc ? aVal - bVal : bVal - aVal;
            } else {
                return asc
                    ? aVal.localeCompare(bVal)
                    : bVal.localeCompare(aVal);
            }
        });
        rows.forEach(r => tbody.appendChild(r));
    }

    /* ── 7b. Table Search Filter ──────────────────────────── */
    function initTableSearch() {
        const input = document.getElementById('table-search');
        const tbody = document.querySelector('#shap-table tbody');
        if (!input || !tbody) return;

        input.addEventListener('input', () => {
            const q = input.value.toLowerCase();
            tbody.querySelectorAll('tr').forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(q) ? '' : 'none';
            });
        });
    }

    /* ── 7c. CSV Export ───────────────────────────────────── */
    function initCSVExport() {
        const btn   = document.getElementById('export-csv-btn');
        const table = document.getElementById('shap-table');
        if (!btn || !table) return;

        btn.addEventListener('click', () => {
            const rows  = [];
            /* headers */
            const ths   = table.querySelectorAll('thead th');
            rows.push(Array.from(ths).map(th => `"${th.textContent.trim()}"`).join(','));

            /* Systematically extract and format the actual sensor data from each table row so it opens cleanly in Excel */
            table.querySelectorAll('tbody tr').forEach(tr => {
                const cells = Array.from(tr.querySelectorAll('td')).map(td => {
                    return `"${td.textContent.trim().replace(/"/g, '""')}"`;
                });
                rows.push(cells.join(','));
            });

            const csv  = rows.join('\r\n');
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href     = url;
            a.download = 'shap_analysis.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    /* ── 8. Python Analytics Fetch (Matplotlib & Seaborn) ─── */
    async function loadPythonAnalytics() {
        const container = document.getElementById('python-analytics-container');
        if (!container) return;

        try {
            const res = await fetch(`${window.BACKEND_URL || 'http://localhost:8000'}/dataset-analytics`);
            if (!res.ok) throw new Error('Analytics endpoint failed');
            const data = await res.json();
            
            container.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 2rem;">
                    <div>
                        <h4 style="color: var(--amber); margin-bottom: 0.5rem;">Sensor Correlation with System Failure</h4>
                        <img src="${data.correlation_plot}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border-faint);">
                    </div>
                    <div>
                        <h4 style="color: var(--amber); margin-bottom: 0.5rem;">Feature Distributions (Healthy vs Failing)</h4>
                        <img src="${data.distribution_plot}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border-faint);">
                    </div>
                </div>
            `;
        } catch (e) {
            console.error(e);
            container.innerHTML = '<span style="color: var(--danger);">Failed to load Python analytics. Ensure backend is running.</span>';
        }
    }

    /* ── 9. D3.js Custom Visual: Feature Network ──────────── */
    function buildD3Network() {
        const container = document.getElementById('d3-network-container');
        if (!container || typeof d3 === 'undefined') return;

        // Give it exact dimensions to render within the flex container nicely
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 400;

        // Clear existing SVG if any
        container.innerHTML = '';

        const svg = d3.select('#d3-network-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('border-radius', '8px');

        // Central target node
        const nodes = [{ id: 'Failure Risk', group: 0, radius: 25 }];
        const links = [];

        // Add feature nodes
        features.forEach((f, i) => {
            nodes.push({
                id: f.name,
                group: f.direction === 'negative' ? 1 : 2, // 1=Raises risk, 2=Lowers risk
                radius: Math.max(10, Math.min(30, f.impact * 0.5)),
                impact: f.impact
            });
            links.push({
                source: f.name,
                target: 'Failure Risk',
                value: f.impact
            });
        });

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(d => d.radius + 5).iterations(2));

        const link = svg.append("g")
            .attr("stroke", "#ffffff")
            .attr("stroke-opacity", 0.2)
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const drag = simulation => {
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        };

        const node = svg.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", d => d.radius)
            .attr("fill", d => d.group === 0 ? "#f7ba2b" : d.group === 1 ? "#ef4444" : "#22c55e")
            .call(drag(simulation));

        node.append("title")
            .text(d => d.id + (d.impact ? ` (Impact: ${d.impact.toFixed(1)}%)` : ''));

        const labels = svg.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .attr("dy", d => d.radius + 15)
            .attr("text-anchor", "middle")
            .text(d => d.id)
            .style("fill", "#a1a1aa")
            .style("font-size", "0.75rem")
            .style("font-family", "sans-serif")
            .style("pointer-events", "none");

        simulation.on("tick", () => {
            link.attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("cx", d => d.x)
                .attr("cy", d => d.y);
                
            labels.attr("x", d => d.x)
                  .attr("y", d => d.y);
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
        buildWaterfall();
        buildRadar();
        animateBars();
        initSortableTable();
        initTableSearch();
        initCSVExport();
        loadPythonAnalytics();
        // Add a slight delay for D3 to ensure the container is fully rendered and styled
        setTimeout(buildD3Network, 100);
    }

}());
