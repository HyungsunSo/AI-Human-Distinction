/**
 * LIME Visualization Module
 * Renders token-level importance highlighting
 */

class LimeVisualizer {
    constructor() {
        this.container = null;
    }

    /**
     * Render LIME tokens with highlighting
     * @param {HTMLElement} container - Container element
     * @param {string} text - Original text
     * @param {Array} tokens - LIME tokens with scores
     */
    render(container, text, tokens) {
        this.container = container;
        if (!container) return;

        // Create a map of word -> score
        const tokenMap = new Map();
        tokens.forEach(t => {
            tokenMap.set(t.word.toLowerCase(), t.score);
        });

        // Split text into words and render
        const words = text.split(/(\s+)/);
        const html = words.map(word => {
            if (/^\s+$/.test(word)) return word; // Keep whitespace

            const cleanWord = word.replace(/[.,!?;:'"()]/g, '').toLowerCase();
            const score = tokenMap.get(cleanWord);

            if (score !== undefined) {
                const className = this.getTokenClass(score);
                const intensity = this.getIntensity(score);
                return `<span class="lime-token ${className}" style="opacity: ${intensity}" title="Score: ${score.toFixed(3)}">${word}</span>`;
            }
            return `<span class="lime-token neutral">${word}</span>`;
        }).join('');

        container.innerHTML = html;
    }

    /**
     * Get CSS class based on score
     */
    getTokenClass(score) {
        if (score > 0.05) return 'positive';  // AI indicator
        if (score < -0.05) return 'negative'; // Human indicator
        return 'neutral';
    }

    /**
     * Get opacity based on score magnitude
     */
    getIntensity(score) {
        const absScore = Math.abs(score);
        return Math.min(0.4 + absScore * 1.2, 1);
    }

    /**
     * Create a legend element
     */
    createLegend() {
        const legend = document.createElement('div');
        legend.className = 'lime-legend';
        legend.innerHTML = `
            <div class="legend-item">
                <span class="legend-color positive"></span>
                <span>AI Indicator</span>
            </div>
            <div class="legend-item">
                <span class="legend-color negative"></span>
                <span>Human Indicator</span>
            </div>
        `;
        return legend;
    }
}

// Export singleton instance
const limeVisualizer = new LimeVisualizer();
