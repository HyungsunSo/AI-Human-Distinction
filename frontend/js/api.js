/**
 * API Client for AI Text Detector
 * Handles communication with backend + mock data for development
 */

const API_BASE_URL = 'http://localhost:8000';

// Mock mode flag - set to false when backend is ready
const USE_MOCK_DATA = false;

/**
 * Mock data for development
 */
const MOCK_DATA = {
    checkpoints: ['best_model.pt', 'mil_model.pt', 'epoch3.pt'],

    // Sample analysis response
    analyzeResponse: {
        prediction: 'AI',
        confidence: 0.87,
        paragraphs: [
            {
                index: 0,
                text: '인공지능 기술의 발전은 우리 사회 전반에 걸쳐 혁명적인 변화를 가져오고 있습니다. 특히 자연어 처리 분야에서의 발전은 인간과 기계 간의 소통 방식을 근본적으로 바꾸고 있습니다.',
                ai_prob: 0.92,
                importance: 0.15
            },
            {
                index: 1,
                text: '그러나 이러한 기술 발전에는 윤리적 고려가 필수적입니다. AI가 생성한 텍스트와 인간이 작성한 텍스트를 구분하는 것이 점점 어려워지고 있기 때문입니다.',
                ai_prob: 0.65,
                importance: 0.08
            },
            {
                index: 2,
                text: '실제로 많은 연구자들은 AI 탐지 기술의 한계를 지적하고 있습니다. 모델이 학습한 데이터의 편향성, 새로운 AI 모델의 출현 등이 탐지를 어렵게 만드는 요인입니다.',
                ai_prob: 0.78,
                importance: 0.12
            },
            {
                index: 3,
                text: '앞으로 AI 텍스트 탐지 기술은 더욱 정교해질 것으로 예상됩니다. 계층적 분석, 설명 가능한 AI 기법 등이 이 분야의 핵심 연구 주제가 되고 있습니다.',
                ai_prob: 0.85,
                importance: 0.10
            }
        ],
        top_paragraph: {
            index: 0,
            text: '인공지능 기술의 발전은 우리 사회 전반에 걸쳐 혁명적인 변화를 가져오고 있습니다. 특히 자연어 처리 분야에서의 발전은 인간과 기계 간의 소통 방식을 근본적으로 바꾸고 있습니다.',
            ai_prob: 0.92
        },
        lime_result: {
            tokens: [
                { word: '혁명적인', score: 0.45 },
                { word: '근본적으로', score: 0.38 },
                { word: '전반에', score: 0.32 },
                { word: '기술', score: 0.28 },
                { word: '발전', score: 0.25 },
                { word: '특히', score: 0.18 },
                { word: '변화', score: 0.15 },
                { word: '방식', score: 0.12 },
                { word: '가져오고', score: 0.08 },
                { word: '사회', score: -0.05 },
                { word: '인간', score: -0.12 },
                { word: '우리', score: -0.18 },
                { word: '소통', score: -0.22 },
                { word: '있습니다', score: -0.08 }
            ]
        },
        deletion_test: {
            original_prob: 0.92,
            modified_prob: 0.61,
            drop: 0.31,
            reliability: 'high',
            removed_tokens: ['혁명적인', '근본적으로', '전반에', '기술', '발전']
        }
    }
};

/**
 * API Client Class
 */
class APIClient {
    constructor() {
        this.baseURL = API_BASE_URL;
        this.useMock = USE_MOCK_DATA;
    }

    /**
     * Get available checkpoints
     */
    async getCheckpoints() {
        if (this.useMock) {
            return { checkpoints: MOCK_DATA.checkpoints };
        }

        const response = await fetch(`${this.baseURL}/checkpoints`);
        if (!response.ok) throw new Error('Failed to fetch checkpoints');
        return response.json();
    }

    /**
     * Load a specific checkpoint
     */
    async loadCheckpoint(checkpointName) {
        if (this.useMock) {
            return { status: 'loaded', model_name: checkpointName };
        }

        const response = await fetch(`${this.baseURL}/checkpoints/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint_name: checkpointName })
        });
        if (!response.ok) throw new Error('Failed to load checkpoint');
        return response.json();
    }

    /**
     * Analyze text - main analysis endpoint
     */
    async analyze(text) {
        if (this.useMock) {
            // Simulate network delay
            await this.delay(1500);

            // Parse the input text into paragraphs for mock response
            const paragraphs = this.parseTextIntoParagraphs(text);

            // Generate mock response based on input
            return this.generateMockResponse(paragraphs);
        }

        const response = await fetch(`${this.baseURL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) throw new Error('Failed to analyze text');
        return response.json();
    }

    /**
     * Helper: Parse text into paragraphs
     */
    parseTextIntoParagraphs(text) {
        return text
            .split('\n')
            .map(p => p.trim())
            .filter(p => p.length > 30);
    }

    /**
     * Helper: Generate mock response based on input paragraphs
     */
    generateMockResponse(paragraphs) {
        if (paragraphs.length === 0) {
            paragraphs = ['샘플 텍스트입니다.'];
        }

        // Generate random AI probabilities for each paragraph
        const analyzedParagraphs = paragraphs.map((text, index) => ({
            index,
            text,
            ai_prob: Math.random() * 0.5 + 0.4, // 0.4 ~ 0.9
            importance: Math.random() * 0.2
        }));

        // Find top paragraph (highest AI prob)
        const topParagraph = analyzedParagraphs.reduce((max, p) =>
            p.ai_prob > max.ai_prob ? p : max
        );

        // Calculate overall confidence
        const avgProb = analyzedParagraphs.reduce((sum, p) => sum + p.ai_prob, 0) / analyzedParagraphs.length;

        // Generate LIME tokens from top paragraph
        const words = topParagraph.text.split(/\s+/).slice(0, 15);
        const limeTokens = words.map(word => ({
            word: word.replace(/[.,!?]/g, ''),
            score: (Math.random() - 0.3) * 0.8 // -0.24 ~ 0.56
        })).filter(t => t.word.length > 0);

        // Sort by absolute score
        limeTokens.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

        // Get top positive tokens for deletion test
        const positiveTokens = limeTokens
            .filter(t => t.score > 0)
            .slice(0, 5)
            .map(t => t.word);

        const drop = positiveTokens.length > 0 ? Math.random() * 0.3 + 0.1 : 0.05;

        return {
            prediction: avgProb > 0.5 ? 'AI' : 'Human',
            confidence: avgProb,
            paragraphs: analyzedParagraphs,
            top_paragraph: {
                index: topParagraph.index,
                text: topParagraph.text,
                ai_prob: topParagraph.ai_prob
            },
            lime_result: {
                tokens: limeTokens
            },
            deletion_test: {
                original_prob: topParagraph.ai_prob,
                modified_prob: topParagraph.ai_prob - drop,
                drop,
                reliability: drop > 0.2 ? 'high' : drop > 0.1 ? 'medium' : 'low',
                removed_tokens: positiveTokens
            }
        };
    }

    /**
     * Helper: Simulate delay
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Export singleton instance
const api = new APIClient();
