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
                            message = `>> Scanning NetGrid for: "${data.args.query}"...`;
                            break;
                        case 'parse_website':
                            const url = new URL(data.args.url);
                            message = `>> Establishing neural link with ${url.hostname}...`;
                            break;
                        default:
                            message = `>> Initializing ${data.tool_name} protocol...`;
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
                    // Do nothing, wait for next status or content
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
                    console.warn("[ALERT] Unknown protocol detected:", data.type);
            }
            this.scrollToBottom();
        } catch (error) {
            console.error("[SYSTEM FAILURE] Protocol error:", error);
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
            cancelDiv.textContent = ">> Protocol execution terminated. Neural link severed.";
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
        msgDiv.className = "message assistant";
        msgDiv.id = "current-response";
        msgDiv.setAttribute('data-content', '');
        msgDiv.setAttribute('data-time', this.getTimestamp());
        this.messageCount++;
        this.messagesDiv.appendChild(msgDiv);
        this.currentMsg = msgDiv;
        this.userInput.disabled = true;
    }

    showThinking() {
        const loadingDiv = this.createLoadingElement(">> Neural processors engaged. Synthesizing response...");
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
            <span>[NETGRID] ${text}</span>
        `;
        return loadingDiv;
    }

    updateCurrentResponse(content) {
        if (this.currentMsg) {
            const formattedContent = content
                .replace(/\[(\d{2}:\d{2}:\d{2})\]/g, '<span class="text-terminal-blue">[$1]</span>')
                .replace(/^(ERROR:)/gm, '<span class="text-red-500">$1</span>')
                .replace(/^(WARNING:)/gm, '<span class="text-yellow-500">$1</span>')
                .replace(/^(SUCCESS:)/gm, '<span class="text-green-500">$1</span>');
            
            this.currentMsg.innerHTML = marked.parse(this.currentMsg.getAttribute('data-content') + formattedContent);
            this.currentMsg.setAttribute('data-content', this.currentMsg.getAttribute('data-content') + content);
        }
    }

    finalizeResponse() {
        if (this.currentMsg) {
            this.currentMsg.removeAttribute("id");
            this.currentMsg = null;
        }
    }

    handleWebSocketError(error) {
        console.error("[CRITICAL] Neural interface failure:", error);
        this.showError(">> Neural interface disrupted. Critical systems compromised. Initiate manual refresh sequence.");
        this.userInput.disabled = false;
    }

    handleWebSocketClose() {
        console.log("[ALERT] Neural link terminated");
        this.showError(">> Neural link terminated. Connection matrix destabilized. Initiate manual refresh sequence.");
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
        userDiv.className = "message user";
        userDiv.setAttribute('data-time', this.getTimestamp());
        userDiv.textContent = message;
        this.messageCount++;
        this.messagesDiv.appendChild(userDiv);
        this.scrollToBottom();
    }

    showError(message) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "message error";
        errorDiv.style.color = "#ff0000";
        errorDiv.setAttribute('data-time', this.getTimestamp());
        errorDiv.textContent = `[CRITICAL ERROR] ${message}`;
        this.messageCount++;
        this.messagesDiv.appendChild(errorDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        const container = document.getElementById("messageContainer");
        container.scrollTop = container.scrollHeight;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.chatInterface = new ChatInterface();
}); 