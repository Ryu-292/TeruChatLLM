import { pipeline } from '@huggingface/transformers';
import * as webllm from "https://esm.run/@mlc-ai/webllm";


const state = {
    vectorStore: [], // To hold vector representations of text chunks
    chatHistory: [],
    isModelLoaded: false
};

// Create a feature-extraction pipeline
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
let engine; 

// Compute sentence embeddings
const sentences = ['This is an example sentence', 'Each sentence is converted'];
const output = await extractor(sentences, { pooling: 'mean', normalize: true });
console.log(output);
// Tensor {
//   dims: [ 2, 384 ],
//   type: 'float32',
//   data: Float32Array(768) [ 0.04592696577310562, 0.07328180968761444, ... ],
//   size: 768
// }

async function loadLLM() {
    const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
    
    // Create the engine
    engine = await webllm.CreateMLCEngine(selectedModel, {
        initProgressCallback: (report) => {
            statusDisplay.innerText = `Loading AI: ${report.text}`;
        }
    });
    
    state.isModelLoaded = true;
    statusDisplay.innerText = "AI Ready! Ask me about your PDFs.";
}

loadLLM(); 



// Function to extract text from a PDF file
async function extractTextFromPDF(file) {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    let fullText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map(item => item.str).join(" ");
        fullText += pageText + "\n";
    }
    return fullText;
}

// Function to chunk text using a sliding window
function chunkText(text, chunkSize = 500, overlap = 100) {
    const chunks = [];
    let i = 0;

    while (i < text.length) {
        // Grab a slice of text
        const chunk = text.substring(i, i + chunkSize);
        chunks.push(chunk);
        
        // Move the index forward by chunk size MINUS the overlap
        i += (chunkSize - overlap);

        // Safety break if we aren't moving forward
        if (chunkSize <= overlap) break; 
    }
    return chunks;
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let sumA = 0;
    let sumB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        sumA += vecA[i] * vecA[i];
        sumB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(sumA) * Math.sqrt(sumB));
}

async function performSearch(queryText, topK = 3) {
    // 1. Turn the user's question into a math vector
    const output = await extractor(queryText, { pooling: 'mean', normalize: true });
    const queryEmbedding = output.tolist()[0];

    // 2. Calculate similarity for every chunk in our store
    const results = state.vectorStore.map(item => {
        return {
            text: item.text,
            source: item.source,
            score: cosineSimilarity(queryEmbedding, item.embedding)
        };
    });

    // 3. Sort by highest score and take the top N (usually 3 or 5)
    results.sort((a, b) => b.score - a.score);
    
    return results.slice(0, topK);
}
// Integration with the UI
const fileInput = document.getElementById('file-input');
const statusDisplay = document.getElementById('upload-status');
fileInput.addEventListener('change', async (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        statusDisplay.innerText = `Preparing to process ${files.length} file(s)...`;
        
        try {
            // Loop through every file uploaded
            for (const file of files) {
                if (file.type !== "application/pdf") {
                    console.warn(`Skipping non-PDF file: ${file.name}`);
                    continue; 
                }

                statusDisplay.innerText = `Processing: ${file.name}...`;
                
                // 1. Extract
                const rawText = await extractTextFromPDF(file);
                
                // 2. Chunk
                const textChunks = chunkText(rawText);

                // 3. Embed & Store
                for (const chunk of textChunks) {
                    const output = await extractor(chunk, { pooling: 'mean', normalize: true });
                    const embedding = output.tolist()[0]; 

                    state.vectorStore.push({ 
                        text: chunk, 
                        embedding: embedding, 
                        source: file.name // This now correctly tags each chunk with its specific filename
                    });
                }
            }

            statusDisplay.innerText = `Success! Knowledge base now has ${state.vectorStore.length} chunks.`;
            console.log("Current Vector Store:", state.vectorStore);
            
        } catch (error) {
            console.error(error);
            statusDisplay.innerText = "Error processing files.";
        }
    }
});

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('msg');
const chatWindow = document.getElementById('chat');
const tempSlider = document.getElementById('temp-slider');
const sysPromptInput = document.getElementById('system-prompt');

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userQuery = chatInput.value.trim();
    if (!userQuery || !state.isModelLoaded) return;

    // 1. Get relevant chunks from PDFs
    const topChunks = await performSearch(userQuery);
    const contextString = topChunks.map(c => `[From ${c.source}]: ${c.text}`).join("\n\n");

    // 2. Build the System Prompt + Context
    const systemInstruction = { 
        role: "system", 
        content: `${sysPromptInput.value}\n\nRelevant Context from Research Papers:\n${contextString}` 
    };

    // 3. Add the user's current question to history
    state.chatHistory.push({ role: "user", content: userQuery });

    // 4. Construct the full "Context Injected" message list
    // This includes: System Instruction + Context + Chat History
    const messages = [systemInstruction, ...state.chatHistory];

    // 5. Generate response
    const reply = await engine.chat.completions.create({ 
        messages, 
        temperature: parseFloat(tempSlider.value) 
    });

    const botResponse = reply.choices[0].message.content;

    // 6. Save the bot's response to history too!
    state.chatHistory.push({ role: "assistant", content: botResponse });

    // 7. Update UI
    const sources = topChunks.map(c => c.source);
    const uniqueSources = [...new Set(sources)]; // Remove duplicates

    chatWindow.innerHTML += `<div class="msg user">${userQuery}</div>`;
    chatWindow.innerHTML += `
        <div class="msg bot">
            ${botResponse}
            <div style="font-size: 0.6rem; margin-top: 5px; opacity: 0.7; border-top: 1px solid #FFB915;">
                Sources used: ${uniqueSources.join(", ")}
            </div>
        </div>`;
    chatWindow.scrollTop = chatWindow.scrollHeight;
    chatInput.value = "";

    
});

const tempVal = document.getElementById('temp-val');
tempSlider.addEventListener('input', (e) => {
    tempVal.textContent = e.target.value;
});
