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
}

const chartManager = new ChartManager();
