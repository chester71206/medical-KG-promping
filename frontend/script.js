// script.js

// API 配置
const API_BASE_URL = 'http://localhost:5000/api';

// 語言檢測器類
class LanguageDetector {
    static detectLanguage(text) {
        if (!text || text.trim().length === 0) return 'unknown';
        
        const chineseRegex = /[\u4e00-\u9fff\u3400-\u4dbf]/g;
        const englishRegex = /[a-zA-Z]/g;
        
        const chineseMatches = text.match(chineseRegex);
        const englishMatches = text.match(englishRegex);
        
        const chineseCount = chineseMatches ? chineseMatches.length : 0;
        const englishCount = englishMatches ? englishMatches.length : 0;
        
        if (chineseCount > 0) {
            return 'chinese';
        } else if (englishCount > 0) {
            return 'english';
        } else {
            return 'unknown';
        }
    }

    static updateLanguageBadge(language) {
        const badge = document.getElementById('languageBadge');
        badge.className = 'language-badge ' + language;
        switch(language) {
            case 'chinese':
                badge.textContent = '中文 🇨🇳';
                break;
            case 'english':
                badge.textContent = 'English 🇺🇸';
                break;
            default:
                badge.textContent = '未知';
        }
    }
}

// API 服務類
class APIService {
    static async makeRequest(endpoint, method = 'GET', data = null) {
        const url = `${API_BASE_URL}${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }

            return result;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    static async processQuestion(question) {
        return this.makeRequest('/process-question', 'POST', { question });
    }

    static async directGemini(question) {
        return this.makeRequest('/direct-gemini', 'POST', { question });
    }

    static async checkHealth() {
        return this.makeRequest('/health');
    }
}

// 主要醫療問答系統類
class MedicalQASystem {
    constructor() {
        this.currentStep = 0;
        this.totalSteps = 6;
        this.initializeEventListeners();
        this.checkServerHealth();
    }

    async checkServerHealth() {
        try {
            await APIService.checkHealth();
            console.log('Backend server is healthy');
        } catch (error) {
            console.error('Backend server health check failed:', error);
            this.showError('無法連接到後端服務器，請確認服務器是否正在運行');
        }
    }

    initializeEventListeners() {
        const questionInput = document.getElementById('questionInput');
        const form = document.getElementById('questionForm');
        const directGeminiBtn = document.getElementById('directGeminiBtn');

        questionInput.addEventListener('input', (e) => {
            const language = LanguageDetector.detectLanguage(e.target.value);
            LanguageDetector.updateLanguageBadge(language);
            this.autoResizeTextarea(e.target);
        });

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.processQuestion();
        });

        directGeminiBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.directGeminiQuestion();
        });
    }

    async processQuestion() {
        const questionText = document.getElementById('questionInput').value.trim();
        if (!questionText) {
            this.showError('請輸入您的醫療問題');
            return;
        }

        this.startProcessing();
        
        try {
            this.updateStep(1, 'active');
            const result = await APIService.processQuestion(questionText);
            
            if (result.success) {
                await this.simulateProcessingSteps();
                this.showResult(result.data); // 呼叫修正後的 showResult
            } else {
                throw new Error(result.error);
            }

        } catch (error) {
            console.error('Processing error:', error);
            this.showError(`處理過程中發生錯誤: ${error.message}`);
        } finally {
            this.stopProcessing();
        }
    }

    async directGeminiQuestion() {
        const questionText = document.getElementById('questionInput').value.trim();
        if (!questionText) {
            this.showError('請輸入您的醫療問題');
            return;
        }

        this.startDirectProcessing();
        
        try {
            const result = await APIService.directGemini(questionText);
            
            if (result.success) {
                this.showDirectResult(result.data); // 呼叫新增的 showDirectResult
            } else {
                throw new Error(result.error);
            }

        } catch (error) {
            console.error('Direct Gemini error:', error);
            this.showError(`直接問Gemini發生錯誤: ${error.message}`);
        } finally {
            this.stopDirectProcessing();
        }
    }

    async simulateProcessingSteps() {
        const steps = [
            { step: 1, delay: 500, message: '語言檢測與翻譯' },
            { step: 2, delay: 1000, message: '醫療實體提取' },
            { step: 3, delay: 800, message: '實體匹配' },
            { step: 4, delay: 1200, message: '知識圖譜搜索' },
            { step: 5, delay: 1000, message: '路徑分析' },
            { step: 6, delay: 1500, message: '生成醫療建議' }
        ];

        for (let i = 0; i < steps.length; i++) {
            const { step, delay } = steps[i];
            this.updateStep(step, 'active');
            await this.delay(delay);
            this.updateStep(step, 'completed');
            if (i < steps.length - 1) {
                await this.delay(200);
            }
        }
    }

    startProcessing() {
        const submitBtn = document.getElementById('submitBtn');
        const submitText = document.getElementById('submitText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const processingSteps = document.getElementById('processingSteps');

        submitBtn.disabled = true;
        submitBtn.classList.add('loading');
        submitText.textContent = '處理中...';
        loadingSpinner.style.display = 'inline-block';
        processingSteps.style.display = 'block';
        this.resetSteps();
        document.getElementById('resultContainer').style.display = 'none';
        document.getElementById('errorContainer').style.display = 'none';
    }

    stopProcessing() {
        const submitBtn = document.getElementById('submitBtn');
        const submitText = document.getElementById('submitText');
        const loadingSpinner = document.getElementById('loadingSpinner');

        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
        submitText.textContent = '🔬 知識圖譜分析';
        loadingSpinner.style.display = 'none';
    }

    startDirectProcessing() {
        const directBtn = document.getElementById('directGeminiBtn');
        const directText = document.getElementById('directSubmitText');
        const directSpinner = document.getElementById('directLoadingSpinner');
        const submitBtn = document.getElementById('submitBtn');

        directBtn.disabled = true;
        submitBtn.disabled = true;
        directBtn.classList.add('loading');
        directText.textContent = '處理中...';
        directSpinner.style.display = 'inline-block';
        document.getElementById('processingSteps').style.display = 'none';
        document.getElementById('resultContainer').style.display = 'none';
        document.getElementById('errorContainer').style.display = 'none';
    }

    stopDirectProcessing() {
        const directBtn = document.getElementById('directGeminiBtn');
        const directText = document.getElementById('directSubmitText');
        const directSpinner = document.getElementById('directLoadingSpinner');
        const submitBtn = document.getElementById('submitBtn');

        directBtn.disabled = false;
        submitBtn.disabled = false;
        directBtn.classList.remove('loading');
        directText.textContent = '🤖 純粹問Gemini';
        directSpinner.style.display = 'none';
    }

    resetSteps() {
        for (let i = 1; i <= this.totalSteps; i++) {
            const step = document.querySelector(`[data-step="${i}"]`);
            const icon = step.querySelector('.step-icon');
            step.classList.remove('active', 'completed');
            icon.classList.remove('pending', 'active', 'completed');
            icon.classList.add('pending');
            icon.textContent = i;
        }
    }

    updateStep(stepNumber, status) {
        const step = document.querySelector(`[data-step="${stepNumber}"]`);
        const icon = step.querySelector('.step-icon');
        step.classList.remove('active', 'completed');
        icon.classList.remove('pending', 'active', 'completed');
        step.classList.add(status);
        icon.classList.add(status);
        icon.textContent = (status === 'completed') ? '✓' : stepNumber;
    }

    // --- 修正後的 showResult 函式 (已移除重複的) ---
    showResult(data) {
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        const detectedLanguage = document.getElementById('detectedLanguage');
        const extractedEntities = document.getElementById('extractedEntities');
        const matchedEntities = document.getElementById('matchedEntities');
        const visualizationSection = document.getElementById('visualizationSection');
        const visualizationContainer = document.getElementById('visualizationContainer');
        const resultDetails = document.getElementById('resultDetails');

        resultContainer.querySelector('.result-label').textContent = '🎯 醫療建議（知識圖譜分析）：';
        resultDetails.style.display = 'block';

        // 使用 marked.js 將 Markdown 格式的文字轉為 HTML
        if (data.final_answer) {
            resultText.innerHTML = marked.parse(data.final_answer);
        } else {
            resultText.textContent = "未能生成答案。";
        }

        // 顯示分析詳情
        detectedLanguage.textContent = this.getLanguageDisplayName(data.detected_language);
        extractedEntities.textContent = data.extracted_entities ? data.extracted_entities.join(', ') : '無';
        matchedEntities.textContent = data.matched_entities ? data.matched_entities.join(', ') : '無';

        // 處理視覺化圖檔
        visualizationContainer.innerHTML = '';
        visualizationSection.style.display = 'none';

        if (data.visualization_url) {
            visualizationSection.style.display = 'block';
            const iframe = document.createElement('iframe');
            iframe.src = data.visualization_url;
            iframe.height = '500px';
            iframe.style.width = '100%';
            iframe.style.border = 'none';
            visualizationContainer.appendChild(iframe);
        }

        resultContainer.style.display = 'block';
        setTimeout(() => resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }

    // --- 新增的 showDirectResult 函式 ---
    showDirectResult(data) {
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');

        // 隱藏知識圖譜分析的特定區塊
        document.getElementById('visualizationSection').style.display = 'none';
        document.getElementById('resultDetails').style.display = 'none';

        // 更新結果標題
        resultContainer.querySelector('.result-label').textContent = '🤖 Gemini 直接回覆：';

        // 同樣使用 marked.js 將 Markdown 格式的文字轉為 HTML
        if (data.final_answer) {
            resultText.innerHTML = marked.parse(data.final_answer);
        } else {
            resultText.textContent = "未能生成答案。";
        }

        resultContainer.style.display = 'block';
        setTimeout(() => resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }


    getLanguageDisplayName(language) {
        switch(language) {
            case 'chinese': return '中文 🇨🇳';
            case 'english': return 'English 🇺🇸';
            default: return '未知';
        }
    }

    showError(message) {
        const errorContainer = document.getElementById('errorContainer');
        const errorText = document.getElementById('errorText');
        errorText.textContent = message;
        errorContainer.style.display = 'block';
        setTimeout(() => errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        const minHeight = 120;
        const maxHeight = 400;
        let newHeight = Math.max(minHeight, textarea.scrollHeight);
        newHeight = Math.min(maxHeight, newHeight);
        textarea.style.height = newHeight + 'px';
        textarea.style.overflowY = (textarea.scrollHeight > maxHeight) ? 'auto' : 'hidden';
    }
}

// 初始化系統
document.addEventListener('DOMContentLoaded', () => {
    new MedicalQASystem();
    addUIEnhancements();
});

// UI 增強功能
function addUIEnhancements() {
    const questionInput = document.getElementById('questionInput');
    
    questionInput.addEventListener('focus', () => questionInput.parentElement.classList.add('focused'));
    questionInput.addEventListener('blur', () => questionInput.parentElement.classList.remove('focused'));

    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.getElementById('questionForm');
            const submitBtn = document.getElementById('submitBtn');
            if (!submitBtn.disabled) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    });

    const originalPlaceholder = questionInput.placeholder;
    questionInput.addEventListener('focus', () => {
        if (questionInput.value === '') {
            questionInput.placeholder = '提示：您可以使用 Ctrl+Enter 快速提交問題';
        }
    });
    questionInput.addEventListener('blur', () => {
        questionInput.placeholder = originalPlaceholder;
    });
}

// 全局錯誤處理
window.addEventListener('error', (e) => console.error('Global error:', e.error));
window.addEventListener('unhandledrejection', (e) => console.error('Unhandled promise rejection:', e.reason));