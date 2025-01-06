const UI_STRINGS = {
    // Header text
    APP_TITLE: 'AUTONOMOUS AGENT NETWORK',
    METRIC_PRIMARY: 'NODES:',
    STATUS_PRIMARY: 'LEARNING: ENABLED',
    MODEL_INFO: 'MODEL: klusterai/Meta-Llama-3.1-405B-Instruct-Turbo',

    // Input placeholders
    INPUT_PLACEHOLDER: 'initiate distributed query sequence...',
    
    // Button text
    BUTTON_SUBMIT: 'EXECUTE',
    BUTTON_CANCEL: 'HALT',

    // Message prefixes
    PREFIX_USER: '[USER]> ',
    PREFIX_SYSTEM: '[SYS]> ',
    PREFIX_STATUS: '[NETGRID] >> ',
    PREFIX_ERROR: '[CRITICAL ERROR] ',

    // Status messages
    STATUS_PROCESSING: 'neural processors engaged. synthesizing response...',
    STATUS_ERROR_CONNECTION: 'neural interface disrupted. critical systems compromised. initiate manual refresh sequence.',
    STATUS_ERROR_TERMINATED: 'neural link terminated. connection matrix destabilized. initiate manual refresh sequence.',
    ERROR_PROTOCOL: 'protocol error:',
    ERROR_SYSTEM: 'system failure:',
    STATUS_CANCELLED: 'protocol execution terminated. neural link severed.',
    STATUS_SEARCHING: 'infiltrating global datastreams',
    STATUS_CONNECTING: 'establishing neural link...',
    STATUS_INITIALIZING: 'initializing {0} protocol...',
    ERROR_UNKNOWN: 'unknown protocol detected:',

    // Metric labels
    METRIC_1_LABEL: 'NODES',
    METRIC_1_FORMAT: '{0}/1,024 ACTIVE',
    METRIC_2_LABEL: 'THROUGHPUT',
    METRIC_2_FORMAT: '{0}% OPTIMAL',
    METRIC_3_LABEL: 'BLOCKS',
    METRIC_4_LABEL: 'ACTIVE QUERIES',
    METRIC_5_LABEL: 'ACCURACY',
    METRIC_6_LABEL: 'CONSENSUS',
    METRIC_PERCENTAGE: '{0}%',

    // Section headers
    SECTION_HEADER_1: 'NEURAL METRICS',
    SECTION_HEADER_2: 'AGENT METRICS',
    SECTION_HEADER_3: 'ACTIVE PROTOCOLS',

    // Status list
    STATUS_ITEM_1: '[ DISTRIBUTED INFERENCE ]',
    STATUS_ITEM_2: '[ PROOF OF INFERENCE ]',
    STATUS_ITEM_3: '[ TURBO MODE ]',
};

// Helper function to format strings with parameters
function formatString(str, ...args) {
    return str.replace(/{(\d+)}/g, (match, index) => {
        return typeof args[index] !== 'undefined' ? args[index] : match;
    });
}

export { UI_STRINGS, formatString }; 