import { UI_STRINGS, formatString } from './constants.js';

class ChatInterface {
    constructor() {
        this.ws = null;
        this.messagesDiv = document.getElementById("messages");
        this.userInput = document.getElementById("userInput");
        this.cancelButton = document.getElementById("cancelButton");
        this.isProcessing = false;
        this.messageCount = 0;
        this.connectWebSocket();
        this.setupEventListeners();
        this.initializeMetricsUpdate();
        this.initializeUI();
    }

    initializeUI() {
        // Initialize header
        document.getElementById('headerTitle').textContent = UI_STRINGS.APP_TITLE;
        document.getElementById('headerNodes').textContent = UI_STRINGS.METRIC_PRIMARY;
        document.getElementById('headerLearning').textContent = UI_STRINGS.STATUS_PRIMARY;
        document.getElementById('headerModel').textContent = UI_STRINGS.MODEL_INFO;

        // Initialize input and buttons
        this.userInput.placeholder = UI_STRINGS.INPUT_PLACEHOLDER;
        document.getElementById('executeButton').textContent = UI_STRINGS.BUTTON_SUBMIT;
        document.getElementById('cancelButton').textContent = UI_STRINGS.BUTTON_CANCEL;

        // Initialize metric labels
        document.getElementById('neuralMetricsTitle').textContent = UI_STRINGS.SECTION_HEADER_1;
        document.getElementById('metricNodesLabel').textContent = UI_STRINGS.METRIC_1_LABEL;
        document.getElementById('metricThroughputLabel').textContent = UI_STRINGS.METRIC_2_LABEL;
        document.getElementById('metricBlocksLabel').textContent = UI_STRINGS.METRIC_3_LABEL;

        document.getElementById('agentMetricsTitle').textContent = UI_STRINGS.SECTION_HEADER_2;
        document.getElementById('metricQueriesLabel').textContent = UI_STRINGS.METRIC_4_LABEL;
        document.getElementById('metricAccuracyLabel').textContent = UI_STRINGS.METRIC_5_LABEL;
        document.getElementById('metricConsensusLabel').textContent = UI_STRINGS.METRIC_6_LABEL;

        document.getElementById('activeProtocolsTitle').textContent = UI_STRINGS.SECTION_HEADER_3;
        const protocolsList = document.getElementById('protocolsList');
        protocolsList.innerHTML = `
            <div>${UI_STRINGS.STATUS_ITEM_1}</div>
            <div>${UI_STRINGS.STATUS_ITEM_2}</div>
            <div>${UI_STRINGS.STATUS_ITEM_3}</div>
        `;

        // Setup button event listeners
        document.getElementById('executeButton').onclick = () => this.sendMessage();
        document.getElementById('cancelButton').onclick = () => this.cancelRequest();
    }

    connectWebSocket() {
        this.ws = new WebSocket(`ws://${window.location.host}/ws`);
        this.ws.onmessage = this.handleWebSocketMessage.bind(this);
        this.ws.onerror = this.handleWebSocketError.bind(this);
        this.ws.onclose = this.handleWebSocketClose.bind(this);
    }

    setupEventListeners() {
        this.userInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                this.sendMessage();
            }
        });
    }

    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case "start_response":
                    this.isProcessing = true;
                    this.cancelButton.classList.remove("hidden");
                    this.createNewAssistantMessage();
                    this.showThinking();
                    break;
                case "tool_start":
                    let message;
                    switch (data.tool_name) {
                        case 'web_search':
                            message = `${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_SEARCHING}${data.args?.query ? ` for: "${data.args.query}"` : "..."}`;
                            break;
                        case 'parse_website':
                            message = data.description || `${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_CONNECTING}`;
                            break;
                        default:
                            message = data.description || `${UI_STRINGS.PREFIX_STATUS}${formatString(UI_STRINGS.STATUS_INITIALIZING, data.tool_name)}`;
                    }
                    this.updateLoadingStatus(message);
                    break;
                case "stream":
                    if (this.currentMsg.querySelector('.loading-status')) {
                        this.currentMsg.querySelector('.loading-status').remove();
                    }
                    this.updateCurrentResponse(data.content);
                    break;
                case "tool_end":
                    break;
                case "end_response":
                    if (this.currentMsg.querySelector('.loading-status')) {
                        this.currentMsg.querySelector('.loading-status').remove();
                    }
                    this.finalizeResponse();
                    this.userInput.disabled = false;
                    this.isProcessing = false;
                    this.cancelButton.classList.add("hidden");
                    break;
                default:
                    console.warn(`${UI_STRINGS.ERROR_UNKNOWN}`, data.type);
            }
            this.scrollToBottom();
        } catch (error) {
            console.error(`${UI_STRINGS.ERROR_SYSTEM} ${UI_STRINGS.ERROR_PROTOCOL}`, error);
        }
    }

    cancelRequest() {
        if (!this.isProcessing) return;
        
        if (this.ws) {
            this.ws.close();
        }

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

        this.connectWebSocket();
    }

    getTimestamp() {
        const now = new Date();
        return `[${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}]`;
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
        const loadingDiv = this.createLoadingElement(`${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_PROCESSING}`);
        this.currentMsg.appendChild(loadingDiv);
    }

    updateLoadingStatus(text) {
        const oldStatus = this.currentMsg.querySelector('.loading-status');
        if (oldStatus) {
            oldStatus.remove();
        }
        const loadingDiv = this.createLoadingElement(text);
        this.currentMsg.appendChild(loadingDiv);
    }

    createLoadingElement(text) {
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "loading-status flex items-center text-agent-blue";
        loadingDiv.innerHTML = `
            <svg class="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>${text}</span>
        `;
        return loadingDiv;
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

    handleWebSocketError(error) {
        console.error(`${UI_STRINGS.PREFIX_ERROR}${UI_STRINGS.STATUS_ERROR_CONNECTION}`, error);
        this.showError(UI_STRINGS.STATUS_ERROR_CONNECTION);
        this.userInput.disabled = false;
    }

    handleWebSocketClose() {
        console.log(`${UI_STRINGS.PREFIX_STATUS}${UI_STRINGS.STATUS_ERROR_TERMINATED}`);
        this.showError(UI_STRINGS.STATUS_ERROR_TERMINATED);
        this.userInput.disabled = false;
    }

    sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;

        this.addUserMessage(message);
        this.ws.send(message);
        this.userInput.value = "";
    }

    addUserMessage(message) {
        const userDiv = document.createElement("div");
        userDiv.className = "message user mb-4";
        userDiv.setAttribute('data-time', this.getTimestamp());
        userDiv.textContent = message;
        this.messageCount++;
        this.messagesDiv.appendChild(userDiv);
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

    scrollToBottom() {
        const container = document.getElementById("messageContainer");
        container.scrollTop = container.scrollHeight;
    }

    initializeMetricsUpdate() {
        // Initialize metrics with random ranges
        this.metrics = {
            nodes: { min: 980, max: 1024, current: 1024 },
            throughput: { min: 82, max: 98, current: 87 },
            blocks: { min: 847000, max: 848000, current: 847231 },
            queries: { min: 2800, max: 3000, current: 2847 },
            accuracy: { min: 99.2, max: 99.9, current: 99.7 },
            consensus: { min: 97.8, max: 99.2, current: 98.2 }
        };

        // Start periodic updates
        setInterval(() => this.updateMetrics(), 2000);
    }

    updateMetrics() {
        // Update each metric with a new random value within its range
        for (const [key, metric] of Object.entries(this.metrics)) {
            const newValue = this.getNewMetricValue(metric);
            metric.current = newValue;
            
            // Find and update all corresponding elements
            const elements = document.querySelectorAll(`[data-metric="${key}"]`);
            elements.forEach(element => {
                let displayValue = newValue;
                
                // Format based on metric type
                switch(key) {
                    case 'nodes':
                        element.textContent = formatString(UI_STRINGS.METRIC_1_FORMAT, displayValue);
                        break;
                    case 'throughput':
                        element.textContent = formatString(UI_STRINGS.METRIC_2_FORMAT, displayValue);
                        break;
                    case 'blocks':
                    case 'queries':
                        element.textContent = displayValue.toLocaleString();
                        break;
                    case 'accuracy':
                    case 'consensus':
                        element.textContent = formatString(UI_STRINGS.METRIC_PERCENTAGE, displayValue);
                        break;
                }
                
                // Add a brief highlight effect
                element.classList.add('value-update');
                setTimeout(() => element.classList.remove('value-update'), 500);
            });
        }
    }

    getNewMetricValue(metric) {
        if (typeof metric.current === 'number') {
            // Calculate a random change
            const maxChange = (metric.max - metric.min) * 0.1;
            const change = (Math.random() - 0.5) * maxChange;
            
            // Apply the change and ensure it stays within bounds
            let newValue = metric.current + change;
            newValue = Math.max(metric.min, Math.min(metric.max, newValue));
            
            // Round appropriately based on the current value's magnitude
            if (newValue >= 1000) {
                return Math.round(newValue);
            } else if (newValue >= 100) {
                return Math.round(newValue * 10) / 10;
            } else {
                return Math.round(newValue * 100) / 100;
            }
        }
        return metric.current;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.chatInterface = new ChatInterface();
}); 