/**
 * Charts Module for AI Text Detector
 * Dark theme compatible
 */

class ChartManager {
    constructor() {
        this.paragraphChart = null;
        this.importanceChart = null;

        // Dark theme colors
        this.colors = {
            ai: '#f85149',
            aiBg: 'rgba(248, 81, 73, 0.3)',
            human: '#3fb950',
            humanBg: 'rgba(63, 185, 80, 0.3)',
            warning: '#d29922',
            warningBg: 'rgba(210, 153, 34, 0.3)',
            accent: '#58a6ff',
            accentBg: 'rgba(88, 166, 255, 0.3)',
            text: '#8b949e',
            grid: 'rgba(48, 54, 61, 0.5)'
        };

        // Chart.js global defaults for dark theme
        Chart.defaults.color = this.colors.text;
        Chart.defaults.borderColor = this.colors.grid;
    }

    getColorByProb(prob) {
        if (prob >= 0.7) return this.colors.ai;
        if (prob >= 0.5) return this.colors.warning;
        return this.colors.human;
    }

    getBgByProb(prob) {
        if (prob >= 0.7) return this.colors.aiBg;
        if (prob >= 0.5) return this.colors.warningBg;
        return this.colors.humanBg;
    }

    renderParagraphChart(paragraphs, topIndex) {
        const ctx = document.getElementById('paragraphChart');
        if (!ctx) return;

        const labels = paragraphs.map((_, i) => `P${i + 1}`);
        const data = paragraphs.map(p => p.ai_prob);
        const bgColors = paragraphs.map((p, i) =>
            i === topIndex ? this.colors.ai : this.getBgByProb(p.ai_prob)
        );
        const borderColors = paragraphs.map((p, i) =>
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
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${(ctx.raw * 100).toFixed(1)}% AI`
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 1,
                        grid: { color: this.colors.grid },
                        ticks: { callback: (v) => `${(v * 100)}%` }
                    },
                    y: { grid: { display: false } }
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

        const labels = paragraphs.map((_, i) => `P${i + 1}`);
        const data = paragraphs.map(p => p.importance);
        const maxImp = Math.max(...data);

        const bgColors = data.map(v => v === maxImp ? this.colors.aiBg : this.colors.accentBg);
        const borderColors = data.map(v => v === maxImp ? this.colors.ai : this.colors.accent);

        if (this.importanceChart) this.importanceChart.destroy();

        this.importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${(ctx.raw * 100).toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        grid: { color: this.colors.grid },
                        ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }
                    },
                    y: { grid: { display: false } }
                }
            }
        });
    }
}

const chartManager = new ChartManager();
