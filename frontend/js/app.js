/**
 * Main Application Logic for AI Text Detector
 * Handles analysis, highlighting, and UI updates
 */

class AITextDetector {
    constructor() {
        this.currentResult = null;
        this.originalText = '';
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
    }

    bindElements() {
        this.textDisplay = document.getElementById('textDisplay');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.fileInput = document.getElementById('fileInput');
        this.checkpointSelect = document.getElementById('checkpointSelect');
        this.resultsContent = document.getElementById('resultsContent');
        this.emptyState = document.getElementById('emptyState');

        // Stats elements
        this.predictionCard = document.getElementById('predictionCard');
        this.predictionLabel = document.getElementById('predictionLabel');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.keyParagraphNum = document.getElementById('keyParagraphNum');
        this.keyParagraphProb = document.getElementById('keyParagraphProb');
        this.reliabilityIcon = document.getElementById('reliabilityIcon');
        this.reliabilityText = document.getElementById('reliabilityText');
        this.probDrop = document.getElementById('probDrop');

        // LIME elements
        this.limeSubtitle = document.getElementById('limeSubtitle');
        this.limeTokens = document.getElementById('limeTokens');
        this.deletedTokens = document.getElementById('deletedTokens');
    }

    bindEvents() {
        this.analyzeBtn.addEventListener('click', () => this.analyze());
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                this.analyze();
            }
        });

        // Chart click handler
        document.addEventListener('paragraphChartClick', (e) => {
            this.scrollToParagraph(e.detail.index);
        });
    }

    handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            this.textDisplay.textContent = event.target.result;
            this.originalText = event.target.result;
        };
        reader.readAsText(file);
    }

    async analyze() {
        const text = this.textDisplay.innerText.trim();
        if (!text) {
            alert('텍스트를 입력해주세요.');
            return;
        }

        this.originalText = text;
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.textContent = 'Analyzing...';

        try {
            const result = await api.analyze(text);
            this.currentResult = result;
            this.renderResults(result);
            this.highlightParagraphs(result.paragraphs);
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('분석에 실패했습니다.');
        } finally {
            this.analyzeBtn.disabled = false;
            this.analyzeBtn.textContent = 'Analyze';
        }
    }

    renderResults(result) {
        // Show results, hide empty state
        this.resultsContent.classList.remove('hidden');
        this.emptyState.classList.add('hidden');

        // Prediction Badge
        this.predictionLabel.textContent = result.prediction;
        this.confidenceValue.textContent = `${(result.confidence * 100).toFixed(0)}%`;
        this.predictionCard.classList.toggle('human', result.prediction === 'Human');

        // Key Paragraph
        this.keyParagraphNum.textContent = `P${result.top_paragraph.index + 1}`;
        this.keyParagraphProb.textContent = `${(result.top_paragraph.ai_prob * 100).toFixed(0)}% AI`;

        // Reliability
        const reliabilityIcons = { high: '✅', medium: '⚠️', low: '❌' };
        const reliabilityLabels = { high: 'High', medium: 'Medium', low: 'Low' };
        this.reliabilityIcon.textContent = reliabilityIcons[result.deletion_test.reliability];
        this.reliabilityText.textContent = reliabilityLabels[result.deletion_test.reliability];

        // Prob Drop
        const drop = result.deletion_test.drop * 100;
        this.probDrop.textContent = `-${drop.toFixed(0)}%`;

        // Charts
        chartManager.renderParagraphChart(result.paragraphs, result.top_paragraph.index);
        chartManager.renderImportanceChart(result.paragraphs);

        // LIME Section
        this.limeSubtitle.textContent = `Paragraph ${result.top_paragraph.index + 1}`;
        this.renderLimeTokens(result.top_paragraph.text, result.lime_result.tokens);
        this.renderDeletedTokens(result.deletion_test.removed_tokens);
    }

    renderLimeTokens(text, tokens) {
        // Build a map of token scores
        const tokenMap = new Map();
        tokens.forEach(t => tokenMap.set(t.word, t.score));

        // Find max score for normalization
        const maxScore = Math.max(...tokens.map(t => Math.abs(t.score)), 0.1);

        // Split text by whitespace and render in original order
        const words = text.split(/\s+/);
        const html = words.map(word => {
            const cleanWord = word.replace(/[.,!?;:'"()]/g, '');
            const score = tokenMap.get(word) ?? tokenMap.get(cleanWord);

            if (score !== undefined && Math.abs(score) > 0.005) {
                // Calculate opacity based on score (0.1 to 0.95 for maximum visible difference)
                const normalizedScore = Math.abs(score) / maxScore;
                const bgOpacity = 0.1 + normalizedScore * 0.85;
                const textOpacity = 0.5 + normalizedScore * 0.5;

                // AI (positive) = coral/red, Human (negative) = green  
                const color = score > 0
                    ? `rgba(255, 138, 122, ${textOpacity})`
                    : `rgba(142, 245, 176, ${textOpacity})`;
                const bgColor = score > 0
                    ? `rgba(255, 100, 80, ${bgOpacity})`
                    : `rgba(100, 230, 150, ${bgOpacity})`;

                return `<span class="token" style="color: ${color}; background: ${bgColor}; font-weight: 600;" title="Score: ${score.toFixed(3)}">${word}</span>`;
            }

            return `<span class="token neutral">${word}</span>`;
        }).join(' ');

        this.limeTokens.innerHTML = html;
    }

    renderDeletedTokens(tokens) {
        if (tokens.length === 0) {
            this.deletedTokens.innerHTML = '';
            return;
        }

        this.deletedTokens.innerHTML = `
            <div class="deleted-tokens-title">Deleted for reliability test:</div>
            ${tokens.map(t => `<span class="deleted-token">${t}</span>`).join('')}
        `;
    }

    highlightParagraphs(paragraphs) {
        const topIdx = this.currentResult?.top_paragraph?.index ?? -1;

        // Build highlighted paragraphs
        const html = paragraphs.map((p, index) => {
            const probClass = this.getHighlightClass(p.ai_prob);
            const limeClass = index === topIdx ? 'lime-target' : '';
            return `<div class="para-highlight ${probClass} ${limeClass}" data-index="${index}">
                <span class="para-prob">[${(p.ai_prob * 100).toFixed(0)}%]</span> ${this.escapeHtml(p.text)}
            </div>`;
        }).join('');

        this.textDisplay.innerHTML = html;

        // Bind click events
        this.textDisplay.querySelectorAll('.para-highlight').forEach(el => {
            el.addEventListener('click', () => {
                const index = parseInt(el.dataset.index);
                this.activateParagraph(index);
            });
        });
    }

    getHighlightClass(prob) {
        if (prob >= 0.7) return 'ai-high';
        if (prob >= 0.5) return 'ai-medium';
        return 'ai-low';
    }

    activateParagraph(index) {
        // Remove active from all
        this.textDisplay.querySelectorAll('.para-highlight').forEach(el => {
            el.classList.remove('active');
        });

        // Add active to target
        const target = this.textDisplay.querySelector(`[data-index="${index}"]`);
        if (target) {
            target.classList.add('active');
        }

        // Update LIME section if we have data
        if (this.currentResult && this.currentResult.paragraphs[index]) {
            const para = this.currentResult.paragraphs[index];
            this.limeSubtitle.textContent = `Paragraph ${index + 1}`;
            // Note: In production, would fetch LIME for this specific paragraph
        }
    }

    scrollToParagraph(index) {
        const target = this.textDisplay.querySelector(`[data-index="${index}"]`);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            this.activateParagraph(index);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AITextDetector();
});
