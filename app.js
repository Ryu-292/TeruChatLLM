console.log('app.js loading...');

// Load libraries from CDN
const transformersPromise = import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0');
const webllmPromise = import('https://esm.run/@mlc-ai/web-llm');

const state = {
    vectorStore: [],
    chatHistory: [],
    isModelLoaded: false,
    engine: null, // Store engine here
    extractor: null
};

async function init() {
    console.log('init() starting...');
    
    try {
        if (document.readyState === 'loading') {
            await new Promise(r => document.addEventListener('DOMContentLoaded', r));
        }

        // Setup PDF.js Worker (Crucial)
        if (window.pdfjsLib) {
            window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        }

        const statusDisplay = document.getElementById('upload-status');
        const fileInput = document.getElementById('file-input');
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('msg');
        const chatWindow = document.getElementById('chat');
        const tempSlider = document.getElementById('temp-slider');
        const sysPromptInput = document.getElementById('system-prompt');
        const tempVal = document.getElementById('temp-val');

        statusDisplay.innerText = 'Loading ML models...';

        // 1. Load Transformers.js
        const { pipeline, env } = await transformersPromise;
        env.allowLocalModels = false; // Force CDN download
        state.extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        console.log('Embeddings model loaded');

        // 2. Load WebLLM
        const webllm = await webllmPromise;
        const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
        
        statusDisplay.innerText = 'Loading LLM model (this may take a minute)...';
        
        // Using CreateMLCEngine (Main Thread) or createWebWorkerMLCEngine (Worker)
        state.engine = await webllm.CreateMLCEngine(selectedModel, {
            initProgressCallback: (report) => {
                statusDisplay.innerText = `Loading AI: ${Math.round(report.progress * 100)}%`;
                console.log(report.text);
            }
        });

        state.isModelLoaded = true;
        statusDisplay.innerText = "AI Ready! Ask me about your PDFs.";

        // --- PDF Utilities ---
        async function extractTextFromPDF(file) {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            let fullText = "";
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                fullText += textContent.items.map(item => item.str).join(" ") + "\n";
            }
            return fullText;
        }

        function chunkText(text, chunkSize = 500, overlap = 100) {
            const chunks = [];
            let i = 0;
            while (i < text.length) {
                chunks.push(text.substring(i, i + chunkSize));
                i += (chunkSize - overlap);
            }
            return chunks;
        }

        function cosineSimilarity(vecA, vecB) {
            let dotProduct = 0, sumA = 0, sumB = 0;
            for (let i = 0; i < vecA.length; i++) {
                dotProduct += vecA[i] * vecB[i];
                sumA += vecA[i] * vecA[i];
                sumB += vecB[i] * vecB[i];
            }
            return dotProduct / (Math.sqrt(sumA) * Math.sqrt(sumB));
        }

        // --- Event Handlers ---
        fileInput.addEventListener('change', async (e) => {
            const files = e.target.files;
            if (!files.length) return;
            
            statusDisplay.innerText = "Processing PDFs...";
            try {
                for (const file of files) {
                    if (file.type !== "application/pdf") continue;
                    const rawText = await extractTextFromPDF(file);
                    const chunks = chunkText(rawText);
                    
                    for (const chunk of chunks) {
                        const output = await state.extractor(chunk, { pooling: 'mean', normalize: true });
                        state.vectorStore.push({ 
                            text: chunk, 
                            embedding: Array.from(output.data), 
                            source: file.name 
                        });
                    }
                }
                statusDisplay.innerText = `Success! ${state.vectorStore.length} chunks indexed.`;
            } catch (err) {
                console.error(err);
                statusDisplay.innerText = "Error processing PDF.";
            }
        });

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const userQuery = chatInput.value.trim();
            if (!userQuery || !state.isModelLoaded) return;

            // UI Update
            chatWindow.innerHTML += `<div class="msg user"><b>You:</b> ${userQuery}</div>`;
            chatInput.value = '';
            statusDisplay.innerText = "Thinking...";

            // RAG: Search
            const queryOutput = await state.extractor(userQuery, { pooling: 'mean', normalize: true });
            const queryEmb = Array.from(queryOutput.data);
            
            const results = state.vectorStore
                .map(item => ({ ...item, score: cosineSimilarity(queryEmb, item.embedding) }))
                .sort((a, b) => b.score - a.score)
                .slice(0, 3);

            const context = results.map(r => `[Source: ${r.source}] ${r.text}`).join("\n\n");

            // LLM Call
            const messages = [
                { role: "system", content: `${sysPromptInput.value}\n\nContext:\n${context}` },
                ...state.chatHistory,
                { role: "user", content: userQuery }
            ];

            const reply = await state.engine.chat.completions.create({
                messages,
                temperature: parseFloat(tempSlider.value)
            });

            const botResponse = reply.choices[0].message.content;
            state.chatHistory.push({ role: "user", content: userQuery });
            state.chatHistory.push({ role: "assistant", content: botResponse });

            chatWindow.innerHTML += `<div class="msg bot"><b>AI:</b> ${botResponse}</div>`;
            chatWindow.scrollTop = chatWindow.scrollHeight;
            statusDisplay.innerText = "AI Ready!";
        });

    } catch (err) {
        console.error("Init Error:", err);
        document.getElementById('upload-status').innerText = "Fatal Error: " + err.message;
    }
}

init();