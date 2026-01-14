/**
 * Charts Module - Minimal Style
 */

class ChartManager {
    constructor() {
        this.paragraphChart = null;
        this.importanceChart = null;

        // Minimal color palette
        this.colors = {
            ai: '#e55039',
            aiBg: 'rgba(229, 80, 57, 0.6)',
            human: '#78e08f',
            humanBg: 'rgba(120, 224, 143, 0.6)',
            warning: '#f6b93b',
            warningBg: 'rgba(246, 185, 59, 0.6)',
            accent: '#4a90d9',
            accentBg: 'rgba(74, 144, 217, 0.6)',
            text: '#666666',
            grid: 'rgba(255, 255, 255, 0.05)'
        };

        // Minimal chart defaults
        Chart.defaults.color = this.colors.text;
        Chart.defaults.borderColor = 'transparent';
        Chart.defaults.font.family = "'SF Mono', 'Monaco', monospace";
        Chart.defaults.font.size = 10;
    }

    getColorByProb(prob) {
        if (prob >= 0.7) return this.colors.aiBg;
        if (prob >= 0.5) return this.colors.warningBg;
        return this.colors.humanBg;
    }

    renderParagraphChart(paragraphs, topIndex) {
        const ctx = document.getElementById('paragraphChart');
        if (!ctx) return;

        // Dynamic height based on paragraph count
        const barHeight = 24;
        const minHeight = 80;
        const calculatedHeight = Math.max(minHeight, paragraphs.length * barHeight);
        ctx.parentElement.style.height = `${calculatedHeight}px`;

        const labels = paragraphs.map((_, i) => `P${i + 1}`);
        const data = paragraphs.map(p => p.ai_prob);
        const bgColors = paragraphs.map((p, i) =>
            i === topIndex ? this.colors.ai : this.getColorByProb(p.ai_prob)
        );

        if (this.paragraphChart) this.paragraphChart.destroy();

        this.paragraphChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: bgColors,
                    borderWidth: 0,
                    borderRadius: 0,
                    barThickness: 16
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1a1a',
                        titleColor: '#ffffff',
                        bodyColor: '#999999',
                        borderWidth: 0,
                        padding: 8,
                        cornerRadius: 0,
                        callbacks: {
                            label: (ctx) => `${(ctx.raw * 100).toFixed(0)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 1,
                        grid: {
                            color: this.colors.grid,
                            drawBorder: false
                        },
                        ticks: {
                            callback: (v) => `${(v * 100)}%`,
                            maxTicksLimit: 5
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { padding: 4 }
                    }
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        document.dispatchEvent(new CustomEvent('paragraphChartClick', {
                            detail: { index: elements[0].index }
                        }));
                    }
                }
            }
        });
    }

    renderImportanceChart(paragraphs) {
        const ctx = document.getElementById('importanceChart');
        if (!ctx) return;

        // Dynamic height based on paragraph count
        const barHeight = 24;
        const minHeight = 80;
        const calculatedHeight = Math.max(minHeight, paragraphs.length * barHeight);
        ctx.parentElement.style.height = `${calculatedHeight}px`;

        const labels = paragraphs.map((_, i) => `P${i + 1}`);
        const data = paragraphs.map(p => p.importance);
        const maxImp = Math.max(...data);

        const bgColors = data.map(v => v === maxImp ? this.colors.accent : this.colors.accentBg);

        if (this.importanceChart) this.importanceChart.destroy();

        this.importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: bgColors,
                    borderWidth: 0,
                    borderRadius: 0,
                    barThickness: 16
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1a1a',
                        titleColor: '#ffffff',
                        bodyColor: '#999999',
                        borderWidth: 0,
                        padding: 8,
                        cornerRadius: 0,
                        callbacks: {
                            label: (ctx) => `${(ctx.raw * 100).toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        grid: {
                            color: this.colors.grid,
                            drawBorder: false
                        },
                        ticks: {
                            callback: (v) => `${(v * 100).toFixed(0)}%`,
                            maxTicksLimit: 5
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { padding: 4 }
                    }
                }
            }
        });
    }

    renderDistributionCharts(metaAnalysis) {
        const container = document.getElementById('metaChartsGrid');
        if (!container) return;

        // Clear existing charts
        container.innerHTML = '';
        this.distributionCharts = this.distributionCharts || {};

        metaAnalysis.features.forEach(feat => {
            const chartId = `chart_${feat.feature_name}`;

            // Create card
            const card = document.createElement('div');
            card.className = 'meta-chart-card';
            card.innerHTML = `
                <div class="meta-chart-header">
                    <span class="meta-chart-title">${feat.display_name}</span>
                    <span class="p-value-badge ${feat.p_value < 0.05 ? 'warning' : ''}">p = ${feat.p_value.toFixed(3)}</span>
                </div>
                <div class="canvas-container">
                    <canvas id="${chartId}"></canvas>
                </div>
                <div class="meta-insight">${feat.interpretation}</div>
            `;
            container.appendChild(card);

            const ctx = document.getElementById(chartId).getContext('2d');

            // Check if this feature uses log transformation (detect by feature name)
            const logFeatures = ['sent_len_median', 'comma_density', 'repeat_ratio_mean'];
            const isLogTransformed = logFeatures.includes(feat.feature_name);

            // Transform the user's value if needed for proper positioning on the chart
            const displayValue = isLogTransformed ? Math.log1p(feat.value) : feat.value;

            // Generate distribution data with 3 std range
            const minX = Math.min(feat.human_stats.mean - 3 * feat.human_stats.std, feat.ai_stats.mean - 3 * feat.ai_stats.std, displayValue);
            const maxX = Math.max(feat.human_stats.mean + 3 * feat.human_stats.std, feat.ai_stats.mean + 3 * feat.ai_stats.std, displayValue);

            const humanDist = this._generateNormalDist(feat.human_stats.mean, feat.human_stats.std, minX, maxX);
            const aiDist = this._generateNormalDist(feat.ai_stats.mean, feat.ai_stats.std, minX, maxX);

            if (this.distributionCharts[chartId]) this.distributionCharts[chartId].destroy();

            this.distributionCharts[chartId] = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Human',
                            data: humanDist,
                            borderColor: this.colors.human,
                            backgroundColor: this.colors.humanMuted || 'rgba(120, 224, 143, 0.1)',
                            fill: true,
                            pointRadius: 0,
                            tension: 0.4
                        },
                        {
                            label: 'AI',
                            data: aiDist,
                            borderColor: this.colors.ai,
                            backgroundColor: this.colors.aiMuted || 'rgba(229, 80, 57, 0.1)',
                            fill: true,
                            pointRadius: 0,
                            tension: 0.4
                        },
                        {
                            label: 'Current',
                            data: [{ x: displayValue, y: 0 }, { x: displayValue, y: Math.max(...humanDist.map(d => d.y), ...aiDist.map(d => d.y)) * 0.8 }],
                            borderColor: this.colors.accent,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    },
                    scales: {
                        y: { display: false },
                        x: {
                            type: 'linear',
                            grid: { color: this.colors.grid }
                        }
                    }
                }
            });
        });

        // Overall Interpretation
        const interpretationEl = document.getElementById('overallInterpretation');
        if (interpretationEl) {
            interpretationEl.textContent = metaAnalysis.overall_interpretation;
        }
    }

    _generateNormalDist(mean, std, min, max, steps = 60) {
        const data = [];
        const stepSize = (max - min) / steps;
        for (let x = min; x <= max; x += stepSize) {
            const y = (1 / (std * Math.sqrt(2 * Math.PI))) *
                Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
            data.push({ x, y });
        }
        return data;
    }
}

const chartManager = new ChartManager();
