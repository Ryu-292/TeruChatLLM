console.log('Terubot loading...');

const transformersPromise = import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0');
const webllmPromise = import('https://esm.run/@mlc-ai/web-llm');

const state = {
    vectorStore: [],
    discussions: [],      
    activeIdx: 0,
    trackedPDFs: [], 
    isModelLoaded: false,
    engine: null,
    extractor: null
};

state.discussions.push({ title: "New Conversation", history: [] });

async function init() {
    const els = {
        status: document.getElementById('upload-status'),
        fileInput: document.getElementById('file-input'),
        chatForm: document.getElementById('chat-form'),
        chatInput: document.getElementById('msg'),
        chatWindow: document.getElementById('chat'),
        tempSlider: document.getElementById('temp-slider'),
        tempVal: document.getElementById('temp-val'),
        sysPrompt: document.getElementById('system-prompt'),
        memoryList: document.getElementById('memory-list'),
        litReviewBtn: document.getElementById('lit-review-btn'),
        quickSummaryBtn: document.getElementById('quick-summary-btn')
    };

    const newChatBtn = document.createElement('button');
    newChatBtn.innerText = "+ New Discussion";
    newChatBtn.className = "new-chat-btn"; 
    if (els.memoryList) els.memoryList.parentNode.insertBefore(newChatBtn, els.memoryList);

    try {
        els.status.innerText = "Loading AI Core...";
        const { pipeline, env } = await transformersPromise;
        env.allowLocalModels = false;
        state.extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

        const webllm = await webllmPromise;
        state.engine = await webllm.CreateMLCEngine("Llama-3.2-1B-Instruct-q4f16_1-MLC", {
            initProgressCallback: (p) => {
                els.status.innerText = `AI Loading: ${Math.round(p.progress * 100)}%`;
            }
        });

        state.isModelLoaded = true;
        els.status.innerText = "Terubot Ready!";
        // Connect temperature slider to display and initialize value
        if (els.tempSlider && els.tempVal) {
            els.tempVal.innerText = els.tempSlider.value;
            els.tempSlider.addEventListener('input', (e) => {
                els.tempVal.innerText = e.target.value;
            });
        }
        updateSidebar();

        function cosineSimilarity(vecA, vecB) {
            let dot = 0, sA = 0, sB = 0;
            for(let i=0; i<vecA.length; i++){
                dot += vecA[i] * vecB[i];
                sA += vecA[i] * vecA[i];
                sB += vecB[i] * vecB[i];
            }
            return dot / (Math.sqrt(sA) * Math.sqrt(sB));
        }

        async function processAndGenerate(file, mode) {
            if (!state.isModelLoaded) return;
            if (state.trackedPDFs.length >= 10) window.deletePDFHandler(state.trackedPDFs[0].filename);
            
            els.status.innerText = `Reading ${file.name}...`;
            const arrayBuffer = await file.arrayBuffer();
            const pdfData = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            // Try to read PDF metadata (e.g., author)
            let author = null;
            try {
                const meta = await pdfData.getMetadata();
                author = (meta && meta.info && meta.info.Author) ? meta.info.Author : null;
            } catch (e) {
                // metadata not available; continue
            }
            let text = "";
            for (let i = 1; i <= pdfData.numPages; i++) {
                const page = await pdfData.getPage(i);
                const content = await page.getTextContent();
                text += content.items.map(s => s.str).join(" ") + " ";
            }

            const pdfHeader = text.substring(0, 1000).replace(/\n/g, ' ');
            const chunks = text.match(/.{1,500}/g) || [];
            const chunkIndices = [];
            
            for (const chunk of chunks) {
                const output = await state.extractor(chunk, { pooling: 'mean', normalize: true });
                state.vectorStore.push({ text: chunk, embedding: Array.from(output.data), source: file.name });
                chunkIndices.push(state.vectorStore.length - 1);
            }

            state.trackedPDFs.push({ filename: file.name, chunkIndices, header: pdfHeader, author });
            updateSidebar();

            if (mode === 'none') return;

            const botDiv = document.createElement('div');
            botDiv.className = `msg bot ${mode}`;
            botDiv.innerHTML = `<b>${mode === 'review' ? '📚 Review' : '📋 Summary'}:</b><br><br><span class="content">...</span>`;
            els.chatWindow.appendChild(botDiv);
            const contentSpan = botDiv.querySelector('.content');

            const reply = await state.engine.chat.completions.create({
                messages: [
                    { role: "system", content: mode === 'review' ? "Provide a literature review of the text." : "Summarize the text in 5 bullet points." },
                    { role: "user", content: `Context:\n${text.substring(0, 6000)}` }
                ],
                stream: true,
                temperature: parseFloat(els.tempSlider?.value) || 0.7
            });

            let full = "";
            for await (const chunk of reply) {
                full += chunk.choices[0]?.delta.content || "";
                contentSpan.innerText = full;
                els.chatWindow.scrollTop = els.chatWindow.scrollHeight;
            }

            state.discussions[state.activeIdx].history.push(
                { role: "user", content: `${mode} for ${file.name}` },
                { role: "assistant", content: full }
            );
            els.status.innerText = "Terubot Ready!";
        }

        function updateSidebar() {
            if (!els.memoryList) return;
            els.memoryList.innerHTML = '<h4>Recent Chats</h4>';
            state.discussions.forEach((d, i) => {
                const item = document.createElement('div');
                item.className = `memory-item ${i === state.activeIdx ? 'active' : ''}`;
                item.innerText = d.title;
                item.onclick = () => { state.activeIdx = i; loadActiveChat(); };
                els.memoryList.appendChild(item);
            });
            const pdfHeader = document.createElement('h4');
            pdfHeader.innerText = "PDF Bank";
            pdfHeader.style.marginTop = "30px";
            els.memoryList.appendChild(pdfHeader);
            state.trackedPDFs.forEach(pdf => {
                const item = document.createElement('div');
                item.className = 'memory-item pdf-entry';
                item.innerHTML = `<span>${pdf.filename}</span><span onclick="window.deletePDFHandler('${pdf.filename.replace(/'/g, "\\'")}')" style="cursor:pointer;color:#ff4d4d">×</span>`;
                els.memoryList.appendChild(item);
            });
        }

        function loadActiveChat() {
            els.chatWindow.innerHTML = '';
            state.discussions[state.activeIdx].history.forEach(m => appendMsg(m.role === 'user' ? 'You' : 'Terubot', m.content, m.role));
            updateSidebar();
        }

        function appendMsg(sender, text, role) {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            div.innerHTML = `<b>${sender}:</b> ${text}`;
            els.chatWindow.appendChild(div);
            els.chatWindow.scrollTop = els.chatWindow.scrollHeight;
        }

        window.deletePDFHandler = (filename) => {
            const idx = state.trackedPDFs.findIndex(p => p.filename === filename);
            if (idx === -1) return;
            state.trackedPDFs[idx].chunkIndices.sort((a,b)=>b-a).forEach(i => state.vectorStore.splice(i,1));
            state.trackedPDFs.splice(idx,1);
            updateSidebar();
        };

        els.chatForm.onsubmit = async (e) => {
            e.preventDefault();
            const query = els.chatInput.value.trim();
            if (!query || !state.isModelLoaded) return;
            appendMsg('You', query, 'user');
            els.chatInput.value = '';

            // If the user asks for the author of a specific PDF, answer directly from metadata when available
            const filenameMatch = query.match(/[\w\-]+\.pdf/i);
            if (filenameMatch) {
                const target = filenameMatch[0].toLowerCase();
                const pdfMeta = state.trackedPDFs.find(p => p.filename.toLowerCase() === target);
                if (pdfMeta && pdfMeta.author) {
                    const direct = `Author of ${pdfMeta.filename}: ${pdfMeta.author}`;
                    appendMsg('Terubot', direct, 'bot');
                    const activeDisc = state.discussions[state.activeIdx];
                    activeDisc.history.push({ role: 'user', content: query }, { role: 'assistant', content: direct });
                    els.status.innerText = "Terubot Ready!";
                    return;
                }
            }

            const hasDocs = state.vectorStore.length > 0;
            let searchResults = [];
            let specificChunks = "";
            let usedSources = [];
            let contextUsed = false;

            if (hasDocs) {
                const qOut = await state.extractor(query, { pooling: 'mean', normalize: true });
                const qEmb = Array.from(qOut.data);
                searchResults = state.vectorStore
                    .map(item => ({ ...item, score: cosineSimilarity(qEmb, item.embedding) }))
                    .sort((a,b) => b.score - a.score)
                    .slice(0, 5);
                const topScore = searchResults[0]?.score ?? 0;
                const THRESHOLD = 0.25;
                if (topScore >= THRESHOLD) {
                    specificChunks = searchResults.map(r => `[${r.source}] ${r.text}`).join("\n");
                    usedSources = [...new Set(searchResults.map(r => r.source))];
                    contextUsed = true;
                }
            }

            const activeDisc = state.discussions[state.activeIdx];
            const docsList = state.trackedPDFs.map(pdf => `- ${pdf.filename}`).join("\n");
            const systemPrompt = !hasDocs ?
                `You are a helpful assistant. Answer conversationally and helpfully.` :
                (contextUsed
                    ? `You are a helpful assistant. Prioritize the provided CONTEXT from the uploaded documents. Be concise and cite filenames used.`
                    : `You are a helpful assistant. No strong matches were found in the uploaded documents; answer based on general knowledge. Avoid claiming confidentiality.`);

            const messages = [
                { role: "system", content: hasDocs && contextUsed ? `${systemPrompt}\n\nDOCS:\n${docsList}\n\nCONTEXT:\n${specificChunks}` : systemPrompt },
                ...activeDisc.history,
                { role: "user", content: query }
            ];

            els.status.innerText = "Thinking...";
            const reply = await state.engine.chat.completions.create({ messages, stream: true, temperature: parseFloat(els.tempSlider?.value) || 0.7 });

            const botDiv = document.createElement('div');
            botDiv.className = "msg bot";
            botDiv.innerHTML = `<b>Terubot:</b> <span class="content"></span>`;
            els.chatWindow.appendChild(botDiv);
            const contentSpan = botDiv.querySelector('.content');

            let full = "";
            for await (const chunk of reply) {
                full += chunk.choices[0]?.delta.content || "";
                contentSpan.innerText = full;
                els.chatWindow.scrollTop = els.chatWindow.scrollHeight;
            }
            // Add citations under the answer only when context was used
            if (contextUsed && usedSources.length) {
                const citeDiv = document.createElement('div');
                citeDiv.className = 'citations';
                citeDiv.innerHTML = `Sources: ${usedSources.map(s => `<span class="cite">${s}</span>`).join(' ')}`;
                botDiv.appendChild(citeDiv);
            }
            activeDisc.history.push({ role: "user", content: query }, { role: "assistant", content: full });
            if (activeDisc.history.length === 2) { activeDisc.title = query.substring(0, 25) + "..."; updateSidebar(); }
            els.status.innerText = "Terubot Ready!";
        };

        newChatBtn.onclick = () => {
            if (state.discussions.length >= 10) state.discussions.shift();
            state.discussions.push({ title: "New Conversation", history: [] });
            state.activeIdx = state.discussions.length - 1;
            loadActiveChat();
        };

        els.litReviewBtn.onclick = () => {
            const p = document.createElement('input'); p.type = 'file'; p.accept = '.pdf';
            p.onchange = (e) => processAndGenerate(e.target.files[0], 'review');
            p.click();
        };

        els.quickSummaryBtn.onclick = () => {
            const p = document.createElement('input'); p.type = 'file'; p.accept = '.pdf';
            p.onchange = (e) => processAndGenerate(e.target.files[0], 'summary');
            p.click();
        };

        els.fileInput.onchange = async (e) => {
            for (const file of e.target.files) await processAndGenerate(file, 'none');
        };

    } catch (err) {
        console.error(err);
    }
}

init();