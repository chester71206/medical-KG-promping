// API 配置
const API_BASE_URL = 'http://localhost:5000/api';

// 語言檢測器類
class LanguageDetector {
    static detectLanguage(text) {
        if (!text || text.trim().length === 0) return 'unknown';
        
        // 檢測中文字符（包含繁體和簡體）
        const chineseRegex = /[\u4e00-\u9fff\u3400-\u4dbf]/g;
        const englishRegex = /[a-zA-Z]/g;
        
        const chineseMatches = text.match(chineseRegex);
        const englishMatches = text.match(englishRegex);
        
        const chineseCount = chineseMatches ? chineseMatches.length : 0;
        const englishCount = englishMatches ? englishMatches.length : 0;
        
        // 只要有任何中文字符就判定為中文（不管英文字符多少）
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

    static async detectLanguage(text) {
        return this.makeRequest('/detect-language', 'POST', { text });
    }

    static async translateText(text, targetLanguage) {
        return this.makeRequest('/translate', 'POST', { 
            text, 
            target_language: targetLanguage 
        });
    }

    static async processQuestion(question) {
        return this.makeRequest('/process-question', 'POST', { 
            question
        });
    }

    static async directGemini(question) {
        return this.makeRequest('/direct-gemini', 'POST', { 
            question
        });
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
        this.voiceRecognition = null;
        this.isListening = false;
        this.initializeVoiceRecognition();
        this.initializeEventListeners();
        this.checkServerHealth();
    }

    // 初始化語音識別
    initializeVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.voiceRecognition = new SpeechRecognition();
            
            this.voiceRecognition.continuous = false;
            this.voiceRecognition.interimResults = false;
            this.voiceRecognition.lang = 'zh-TW';
            
            this.voiceRecognition.onstart = () => {
                this.isListening = true;
                document.getElementById('voiceBtn').classList.add('listening');
                console.log('🎤 開始語音識別');
            };

            this.voiceRecognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                const questionInput = document.getElementById('questionInput');
                questionInput.value += transcript;
                questionInput.dispatchEvent(new Event('input'));
                console.log('📝 識別結果:', transcript);
            };

            this.voiceRecognition.onend = () => {
                this.isListening = false;
                document.getElementById('voiceBtn').classList.remove('listening');
                console.log('🛑 語音識別結束');
            };

            this.voiceRecognition.onerror = (event) => {
                this.isListening = false;
                document.getElementById('voiceBtn').classList.remove('listening');
                console.error('❌ 語音識別錯誤:', event.error);
            };
            
            console.log('✅ 語音識別已初始化');
        } else {
            console.warn('❌ 瀏覽器不支援語音識別');
        }
    }

    // 切換語音輸入
    toggleVoiceInput() {
        if (!this.voiceRecognition) {
            alert('您的瀏覽器不支援語音識別功能');
            return;
        }

        if (this.isListening) {
            this.voiceRecognition.stop();
        } else {
            this.voiceRecognition.start();
        }
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
        const voiceBtn = document.getElementById('voiceBtn');

        // 監聽輸入變化，即時檢測語言並自動調整高度
        questionInput.addEventListener('input', (e) => {
            const language = LanguageDetector.detectLanguage(e.target.value);
            LanguageDetector.updateLanguageBadge(language);
            
            // 自動調整textarea高度
            this.autoResizeTextarea(e.target);
        });

        // 表單提交（知識圖譜分析）
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.processQuestion();
        });

        // 直接問Gemini按鈕
        directGeminiBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.directGeminiQuestion();
        });

        // 語音輸入按鈕
        voiceBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.toggleVoiceInput();
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
            // 調用後端API處理問題
            this.updateStep(1, 'active');
            
            const result = await APIService.processQuestion(questionText);
            
            if (result.success) {
                // 模擬各個步驟的進度更新
                await this.simulateProcessingSteps();
                
                // 顯示結果
                this.showResult(result.data);
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
                // 顯示結果，標註為直接Gemini回答
                this.showDirectResult(result.data);
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
            
            // 如果不是最後一步，添加小延遲
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

        // 重置所有步驟狀態
        this.resetSteps();

        // 隱藏之前的結果
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

        // 禁用兩個按鈕，避免同時執行
        directBtn.disabled = true;
        submitBtn.disabled = true;
        directBtn.classList.add('loading');
        directText.textContent = '處理中...';
        directSpinner.style.display = 'inline-block';

        // 隱藏處理步驟（因為直接問Gemini不需要這些步驟）
        document.getElementById('processingSteps').style.display = 'none';

        // 隱藏之前的結果
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
        
        // 移除所有狀態
        step.classList.remove('active', 'completed');
        icon.classList.remove('pending', 'active', 'completed');
        
        // 添加新狀態
        step.classList.add(status);
        icon.classList.add(status);
        
        if (status === 'completed') {
            icon.textContent = '✓';
        } else {
            icon.textContent = stepNumber;
        }
    }

    showResult(data) {
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        const detectedLanguage = document.getElementById('detectedLanguage');
        const extractedEntities = document.getElementById('extractedEntities');
        const matchedEntities = document.getElementById('matchedEntities');
        
        // 更新結果標題，顯示使用的方法
        const resultLabel = resultContainer.querySelector('.result-label');
        resultLabel.textContent = '🎯 醫療建議（知識圖譜分析）：';
        
        // 顯示主要結果（使用Markdown渲染）
        if (typeof marked !== 'undefined') {
            resultText.innerHTML = marked.parse(data.final_answer);
        } else {
            resultText.textContent = data.final_answer;
        }
        
        // 顯示分析詳情
        detectedLanguage.textContent = this.getLanguageDisplayName(data.detected_language);
        extractedEntities.textContent = data.extracted_entities ? data.extracted_entities.join(', ') : '無';
        matchedEntities.textContent = data.matched_entities ? data.matched_entities.join(', ') : '無';
        
        resultContainer.style.display = 'block';
        
        // 滾動到結果區域
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    showDirectResult(data) {
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        const detectedLanguage = document.getElementById('detectedLanguage');
        const extractedEntities = document.getElementById('extractedEntities');
        const matchedEntities = document.getElementById('matchedEntities');
        
        // 更新結果標題，顯示使用的方法
        const resultLabel = resultContainer.querySelector('.result-label');
        resultLabel.textContent = '🤖 Gemini純粹回答：';
        
        // 顯示主要結果（使用Markdown渲染）
        if (typeof marked !== 'undefined') {
            resultText.innerHTML = marked.parse(data.final_answer);
        } else {
            resultText.textContent = data.final_answer;
        }
        
        // 顯示分析詳情（純粹版）
        detectedLanguage.textContent = '無語言處理（原始輸入）';
        extractedEntities.textContent = '無實體提取（純粹問答）';
        matchedEntities.textContent = '無實體匹配（純粹問答）';
        
        resultContainer.style.display = 'block';
        
        // 滾動到結果區域
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
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
        
        // 滾動到錯誤區域
        setTimeout(() => {
            errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    autoResizeTextarea(textarea) {
        // 重置高度以獲得正確的scrollHeight
        textarea.style.height = 'auto';
        
        // 設置最小和最大高度
        const minHeight = 120; // 最小高度
        const maxHeight = 400; // 最大高度
        
        // 計算新高度
        let newHeight = Math.max(minHeight, textarea.scrollHeight);
        newHeight = Math.min(maxHeight, newHeight);
        
        // 應用新高度
        textarea.style.height = newHeight + 'px';
        
        // 如果超過最大高度，顯示滾動條
        if (textarea.scrollHeight > maxHeight) {
            textarea.style.overflowY = 'auto';
        } else {
            textarea.style.overflowY = 'hidden';
        }
    }
}

// 初始化系統
document.addEventListener('DOMContentLoaded', () => {
    new MedicalQASystem();
    
    // 添加一些額外的用戶體驗增強
    addUIEnhancements();
});

// UI 增強功能
function addUIEnhancements() {
    // 輸入框焦點效果
    const questionInput = document.getElementById('questionInput');
    
    questionInput.addEventListener('focus', () => {
        questionInput.parentElement.classList.add('focused');
    });
    
    questionInput.addEventListener('blur', () => {
        questionInput.parentElement.classList.remove('focused');
    });

    // 鍵盤快捷鍵支持
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter 提交表單
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.getElementById('questionForm');
            const submitBtn = document.getElementById('submitBtn');
            
            if (!submitBtn.disabled) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    });

    // 添加提示文字
    const questionInput2 = document.getElementById('questionInput');
    const originalPlaceholder = questionInput2.placeholder;
    
    questionInput2.addEventListener('focus', () => {
        if (questionInput2.value === '') {
            questionInput2.placeholder = '提示：您可以使用 Ctrl+Enter 快速提交問題';
        }
    });
    
    questionInput2.addEventListener('blur', () => {
        questionInput2.placeholder = originalPlaceholder;
    });
}

// 全局錯誤處理
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

// 未處理的 Promise 拒絕
window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});