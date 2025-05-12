document.addEventListener('DOMContentLoaded', () => {
    const appContainer = document.getElementById('app-container');
    const popup = document.getElementById('app-popup');
    const closePopupBtn = document.getElementById('close-popup');
    const copyPromptBtn = document.getElementById('copy-prompt');
    const showExampleBtn = document.getElementById('show-example');
    const toast = document.getElementById('toast');

    const popupTitle = document.getElementById('popup-title');
    const popupIcon = document.getElementById('popup-icon');
    const popupDescription = document.getElementById('popup-description');
    const popupPrompt = document.getElementById('popup-prompt');
    const exampleBox = document.getElementById('example-box');
    const exampleUser = document.getElementById('example-user');
    const exampleResult = document.getElementById('example-result');

    const toolkitFilters = document.getElementById('toolkit-filters');

    let apps = [];
    const toolkits = new Set();
    let chainMode = false;
    let selectedTemplates = [];


    fetch('/builder/builder.json')
        .then(response => response.json())
        .then(data => {
            apps = Object.values(data);
            renderApps(apps);
        });

    const iconMap = {
        Writing: '📝',
        Research: '🔬',
        SEO: '📈',
        Coding: '💻',
        Learning: '📚',
        Brainstorming: '🧠',
        Productivity: '⚙️',
        Sessions: '🗓️',
        Coaching: '🧘',
        Marketing: '📣',
        Reflection: '🔍',
        Collaboration: '🤝',
        Freestyle_Cognition: '🧩'
    };

    function renderFilters(apps) {
        toolkitFilters.innerHTML = ''; // Clear any old buttons
        toolkits.clear();


        apps.forEach(app => {
            if (app.toolkit) {
                const split = Array.isArray(app.toolkit) ? app.toolkit : app.toolkit.split(',');
                split.forEach(tk => toolkits.add(tk.trim()));
            }
        });

        toolkits.forEach(tk => {
            const btn = document.createElement('button');
            const normalize = str => str.trim().replace(/\s+/g, '_');
            const icon = iconMap[normalize(tk)] || '🔧';
            btn.innerHTML = `<span class="icon">${icon}</span><span class="text">${tk}</span>`;
            btn.className = 'toolkit-button';
            btn.addEventListener('click', () => filterByToolkit(tk));
            toolkitFilters.appendChild(btn);
        });

        const clearBtn = document.createElement('button');
        clearBtn.innerHTML = `<span class="icon">🧹</span><span class="text">Clear Filters</span>`;
        clearBtn.className = 'toolkit-button clear';
        clearBtn.addEventListener('click', () => renderApps(apps));
        toolkitFilters.appendChild(clearBtn);
    }

    function renderApps(appList) {
        appContainer.innerHTML = '';
        appList.forEach(app => {
            const card = document.createElement('div');
            card.className = 'app-card';

            const icon = document.createElement('img');
            icon.src = `/builder/img/${app.slug || 'default'}.png`;
            icon.alt = app.name || 'App Icon';

            const title = document.createElement('h3');
            title.textContent = app.name;

            const description = document.createElement('p');
            description.textContent = app.short_description;

            card.appendChild(icon);
            card.appendChild(title);
            card.appendChild(description);

            card.addEventListener('click', () => {
                console.log("chain mode:", chainMode);
                if (chainMode) {
                    console.log("Chain mode active:", app.name);
                    if (!selectedTemplates.includes(app)) {
                        selectedTemplates.push(app);
                        card.classList.add("selected");
                        renderSelectedChain();
                    }
                } else {
                    console.log("chain mode:", chainMode);
                    openPopup(app); // existing behavior
                }
            });


            appContainer.appendChild(card);
        });
    }

    function openPopup(app) {
        console.log(app);
        popupTitle.textContent = app.name || '';
        popupIcon.src = `/builder/img/${app.slug || 'default'}.png`;
        popupDescription.textContent = app.full_description || '';
        popupPrompt.textContent = app.prompt || '';
        popup.dataset.promptName = app.name || 'Prompt';
        exampleUser.textContent = app.example_user || '';
        exampleResult.textContent = app.example_result || '';
        exampleBox.classList.add('hidden');
        popup.classList.remove('hidden');
        popup.classList.add('visible');
    }

    function filterByToolkit(toolkit) {
        const filtered = apps.filter(app => {
            const values = Array.isArray(app.toolkit) ? app.toolkit : app.toolkit.split(',');
            return values.map(v => v.trim()).includes(toolkit);
        });

        // Remove active class from all buttons
        document.querySelectorAll('.toolkit-button').forEach(btn => btn.classList.remove('active'));

        // Add active class to clicked button
        const activeBtn = Array.from(document.querySelectorAll('.toolkit-button'))
            .find(btn => btn.textContent === toolkit);
        if (activeBtn) activeBtn.classList.add('active');

        renderApps(filtered);
    }

    closePopupBtn.addEventListener('click', () => {
        console.log("Close clicked");
        popup.classList.add('hidden');
        popup.classList.remove('visible');
    });

    showExampleBtn.addEventListener('click', () => {
        exampleBox.classList.toggle('hidden');
    });

    copyPromptBtn.addEventListener('click', () => {
        const promptText = popupPrompt.textContent;
        navigator.clipboard.writeText(promptText).then(() => {
            showToast(`✅ Prompt "${popup.dataset.promptName}" copied!`);
        });
    });

    function showToast(message) {
        toast.textContent = message;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 2000);
    }

    function renderSelectedChain() {
        const list = document.getElementById("selected-prompts-list");
        list.innerHTML = "";
        selectedTemplates.forEach((item, idx) => {
            const li = document.createElement("li");
            li.innerHTML = `${idx + 1}. ${item.name} <button onclick="removeFromChain(${idx})">✖</button>`;
            list.appendChild(li);
        });

        document.getElementById("chain-display").style.display = selectedTemplates.length ? "block" : "none";
    }

    window.removeFromChain = function removeFromChain(index) {
        selectedTemplates.splice(index, 1);
        renderSelectedChain();
    }

    document.getElementById("start-chain").addEventListener("click", () => {
        console.log("Start chain clicked");
        if (chainMode) {
            showToast("Chain mode is already active.");
            return;
        }
        chainMode = true;
        selectedTemplates = [];
        renderSelectedChain();
        document.querySelectorAll(".app-card.selected").forEach(card => {
            card.classList.remove("selected");
        });
    });

    document.getElementById("clear-chain").addEventListener("click", () => {
        chainMode = false;
        selectedTemplates = [];
        renderSelectedChain();

        // Remove visual highlights
        document.querySelectorAll(".app-card.selected").forEach(card => {
            card.classList.remove("selected");
        });
    });

    document.getElementById("generate-chain").addEventListener("click", () => {
        const combined = selectedTemplates.map(t => t.prompt).join("\n\n---\n\n");
        document.getElementById("combined-prompt-output").value = combined;
    });

    document.getElementById("copy-combined-prompt").addEventListener("click", () => {
        const textarea = document.getElementById("combined-prompt-output");
        textarea.select();
        textarea.setSelectionRange(0, 99999); // for mobile support

        navigator.clipboard.writeText(textarea.value).then(() => {
            showToast("✅ Prompt copied to clipboard!");
        }).catch(() => {
            showToast("❌ Failed to copy prompt.");
        });
    });


    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && !popup.classList.contains('hidden')) {
            popup.classList.add('hidden');
            popup.classList.remove('visible');
        }
    });
});

