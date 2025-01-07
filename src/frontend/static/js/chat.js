import { UI_STRINGS, formatString } from './constants.js';

class ChatInterface {
    constructor() {
        this.ws = null;
        this.messagesDiv = document.getElementById("messages");
        this.userInput = document.getElementById("userInput");
        this.cancelButton = document.getElementById("cancelButton");
        this.isProcessing = false;
        this.messageCount = 0;
        this.initializeChat();
    }

    async initializeChat() {
        try {
            const modelInfoResponse = await fetch('/api/models');
            if (!modelInfoResponse.ok) throw new Error('Failed to fetch models');
            
            const models = await modelInfoResponse.json();
            if (!Array.isArray(models) || models.length === 0) throw new Error("No models available");
            
            const currentModel = models[0].id;
            UI_STRINGS.MODEL_INFO = `MODEL: ${currentModel}`;
            
            const headerModel = document.getElementById('headerModel');
            if (models.length > 1) {
                headerModel.innerHTML = `MODEL: <select id="modelSelector" class="bg-transparent border-none text-terminal-green cursor-pointer">
                    ${models.map(model => `<option value="${model.id}" ${model.id === currentModel ? 'selected' : ''}>${model.id}</option>`).join('')}
                </select>`;
                
                document.getElementById('modelSelector').addEventListener('change', (e) => {
                    UI_STRINGS.MODEL_INFO = `MODEL: ${e.target.value}`;
                    if (this.ws) this.ws.close();
                    this.initializeWebSocket();
                });
            } else {
                headerModel.textContent = UI_STRINGS.MODEL_INFO;
            }
            
            this.initializeWebSocket();
            this.setupEventListeners();
            this.initializeMetricsUpdate();
            this.initializeUI();
        } catch (error) {
            console.error('Failed to initialize chat:', error);
            this.showError(error.message);
        }
    }

    initializeWebSocket() {
        const modelSelector = document.getElementById('modelSelector');
        const currentModel = modelSelector ? modelSelector.value : UI_STRINGS.MODEL_INFO.replace('MODEL: ', '');
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        const wsUrlWithModel = currentModel ? `${wsUrl}?model=${encodeURIComponent(currentModel)}` : wsUrl;
        
        this.ws = new WebSocket(wsUrlWithModel);
        this.ws.onmessage = this.handleWebSocketMessage.bind(this);
        this.ws.onerror = () => this.showError(UI_STRINGS.STATUS_ERROR_CONNECTION);
        this.ws.onclose = () => this.showError(UI_STRINGS.STATUS_ERROR_TERMINATED);
    }

    initializeUI() {
        document.getElementById('headerTitle').textContent = UI_STRINGS.APP_TITLE;
        document.getElementById('headerNodes').textContent = UI_STRINGS.METRIC_PRIMARY;
        document.getElementById('headerLearning').textContent = UI_STRINGS.STATUS_PRIMARY;
        this.userInput.placeholder = UI_STRINGS.INPUT_PLACEHOLDER;
        document.getElementById('executeButton').textContent = UI_STRINGS.BUTTON_SUBMIT;
        document.getElementById('cancelButton').textContent = UI_STRINGS.BUTTON_CANCEL;

        const uiElements = {
            'neuralMetricsTitle': UI_STRINGS.SECTION_HEADER_1,
            'metricNodesLabel': UI_STRINGS.METRIC_1_LABEL,
            'metricThroughputLabel': UI_STRINGS.METRIC_2_LABEL,
            'metricBlocksLabel': UI_STRINGS.METRIC_3_LABEL,
            'agentMetricsTitle': UI_STRINGS.SECTION_HEADER_2,
            'metricQueriesLabel': UI_STRINGS.METRIC_4_LABEL,
            'metricAccuracyLabel': UI_STRINGS.METRIC_5_LABEL,
            'metricConsensusLabel': UI_STRINGS.METRIC_6_LABEL,
            'activeProtocolsTitle': UI_STRINGS.SECTION_HEADER_3
        };

        Object.entries(uiElements).forEach(([id, text]) => {
            document.getElementById(id).textContent = text;
        });

        document.getElementById('protocolsList').innerHTML = `
            <div>${UI_STRINGS.STATUS_ITEM_1}</div>
            <div>${UI_STRINGS.STATUS_ITEM_2}</div>
            <div>${UI_STRINGS.STATUS_ITEM_3}</div>
        `;

        document.getElementById('executeButton').onclick = () => this.sendMessage();
        document.getElementById('cancelButton').onclick = () => this.cancelRequest();
    }

    setupEventListeners() {
        this.userInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") this.sendMessage();
        });
    }

    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            const handlers = {
                start_response: () => {
                    this.isProcessing = true;
                    this.cancelButton.classList.remove("hidden");
                    this.createNewAssistantMessage();
                    this.showThinking();
                },
                tool_start: () => {
                    const messages = {
                        web_search: `${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_SEARCHING}${data.args?.query ? ` for: "${data.args.query}"` : "..."}`,
                        parse_website: data.description || `${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_CONNECTING}`,
                        default: data.description || `${UI_STRINGS.PREFIX_STATUS}${formatString(UI_STRINGS.STATUS_INITIALIZING, data.tool_name)}`
                    };
                    this.updateLoadingStatus(messages[data.tool_name] || messages.default);
                },
                stream: () => {
                    this.currentMsg.querySelector('.loading-status')?.remove();
                    this.updateCurrentResponse(data.content);
                },
                end_response: () => {
                    this.currentMsg.querySelector('.loading-status')?.remove();
                    this.finalizeResponse();
                    this.userInput.disabled = false;
                    this.isProcessing = false;
                    this.cancelButton.classList.add("hidden");
                }
            };

            handlers[data.type]?.();
            this.scrollToBottom();
        } catch (error) {
            console.error(`${UI_STRINGS.ERROR_SYSTEM} ${UI_STRINGS.ERROR_PROTOCOL}`, error);
        }
    }

    sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;

        const userDiv = document.createElement("div");
        userDiv.className = "message user mb-4";
        userDiv.setAttribute('data-time', this.getTimestamp());
        userDiv.textContent = message;
        this.messageCount++;
        this.messagesDiv.appendChild(userDiv);
        
        this.ws.send(message);
        this.userInput.value = "";
        this.scrollToBottom();
    }

    showError(message) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "message error mb-4";
        errorDiv.style.color = "#ff0000";
        errorDiv.setAttribute('data-time', this.getTimestamp());
        errorDiv.textContent = `${UI_STRINGS.PREFIX_ERROR}${message}`;
        this.messageCount++;
        this.messagesDiv.appendChild(errorDiv);
        this.scrollToBottom();
    }

    createNewAssistantMessage() {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message assistant mb-4 whitespace-pre-wrap";
        msgDiv.id = "current-response";
        msgDiv.setAttribute('data-content', '');
        msgDiv.setAttribute('data-time', this.getTimestamp());
        this.messageCount++;
        this.messagesDiv.appendChild(msgDiv);
        this.currentMsg = msgDiv;
        this.userInput.disabled = true;
    }

    showThinking() {
        this.updateLoadingStatus(`${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_PROCESSING}`);
    }

    updateLoadingStatus(text) {
        this.currentMsg.querySelector('.loading-status')?.remove();
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "loading-status flex items-center text-agent-blue";
        loadingDiv.innerHTML = `
            <svg class="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>${text}</span>
        `;
        this.currentMsg.appendChild(loadingDiv);
    }

    updateCurrentResponse(content) {
        if (this.currentMsg) {
            const existingContent = this.currentMsg.getAttribute('data-content') || '';
            const newContent = existingContent + content;
            
            const formattedContent = newContent
                .replace(/\[(\d{2}:\d{2}:\d{2})\]/g, '<span class="text-terminal-blue">[$1]</span>')
                .replace(/^(ERROR:)/gm, '<span class="text-red-500">$1</span>')
                .replace(/^(WARNING:)/gm, '<span class="text-yellow-500">$1</span>')
                .replace(/^(SUCCESS:)/gm, '<span class="text-green-500">$1</span>')
                .replace(/\n\n/g, '<br><br>')
                .replace(/\n/g, '<br>');
            
            this.currentMsg.innerHTML = formattedContent;
            this.currentMsg.setAttribute('data-content', newContent);
        }
    }

    finalizeResponse() {
        if (this.currentMsg) {
            this.currentMsg.removeAttribute("id");
            this.currentMsg = null;
        }
    }

    cancelRequest() {
        if (!this.isProcessing) return;
        
        if (this.ws) this.ws.close();

        if (this.currentMsg) {
            this.currentMsg.querySelector('.loading-status')?.remove();
            const cancelDiv = document.createElement("div");
            cancelDiv.className = "text-error-red text-sm mt-2";
            cancelDiv.textContent = `${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_CANCELLED}`;
            this.currentMsg.appendChild(cancelDiv);
            this.finalizeResponse();
        }

        this.isProcessing = false;
        this.userInput.disabled = false;
        this.cancelButton.classList.add("hidden");
        this.initializeWebSocket();
    }

    getTimestamp() {
        const now = new Date();
        return `[${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}]`;
    }

    scrollToBottom() {
        document.getElementById("messageContainer").scrollTop = document.getElementById("messageContainer").scrollHeight;
    }

    initializeMetricsUpdate() {
        this.metrics = {
            nodes: { min: 980, max: 1024, current: 1024 },
            throughput: { min: 82, max: 98, current: 87 },
            blocks: { min: 847000, max: 848000, current: 847231 },
            queries: { min: 2800, max: 3000, current: 2847 },
            accuracy: { min: 99.2, max: 99.9, current: 99.7 },
            consensus: { min: 97.8, max: 99.2, current: 98.2 }
        };
        setInterval(() => this.updateMetrics(), 2000);
    }

    updateMetrics() {
        Object.entries(this.metrics).forEach(([key, metric]) => {
            const maxChange = (metric.max - metric.min) * 0.1;
            let newValue = metric.current + (Math.random() - 0.5) * maxChange;
            newValue = Math.max(metric.min, Math.min(metric.max, newValue));
            
            if (newValue >= 1000) newValue = Math.round(newValue);
            else if (newValue >= 100) newValue = Math.round(newValue * 10) / 10;
            else newValue = Math.round(newValue * 100) / 100;
            
            metric.current = newValue;
            
            document.querySelectorAll(`[data-metric="${key}"]`).forEach(element => {
                element.textContent = key === 'nodes' ? formatString(UI_STRINGS.METRIC_1_FORMAT, newValue)
                    : key === 'throughput' ? formatString(UI_STRINGS.METRIC_2_FORMAT, newValue)
                    : ['blocks', 'queries'].includes(key) ? newValue.toLocaleString()
                    : formatString(UI_STRINGS.METRIC_PERCENTAGE, newValue);
                
                element.classList.add('value-update');
                setTimeout(() => element.classList.remove('value-update'), 500);
            });
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.chatInterface = new ChatInterface();
}); 