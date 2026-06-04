const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');

searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }

    loading.classList.remove('hidden');
    error.classList.add('hidden');
    results.innerHTML = '';

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 10 })
        });

        if (!response.ok) {
            throw new Error('Search failed');
        }

        const data = await response.json();
        displayResults(data.results);
    } catch (err) {
        showError('Search failed: ' + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

function displayResults(items) {
    results.innerHTML = '';

    if (items.length === 0) {
        results.innerHTML = '<p style="color: white; text-align: center;">No results found</p>';
        return;
    }

    items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        card.innerHTML = `
            <img src="${item.image_url}" alt="${item.image_id}">
            <div class="result-info">
                <div class="rank">Rank #${item.rank}</div>
                <div class="score">Score: ${item.score.toFixed(4)}</div>
                <div class="image-id">${item.image_id}</div>
                <button class="find-similar-btn" data-image-id="${item.image_id}">Find Similar</button>
            </div>
        `;
        
        results.appendChild(card);
    });

    document.querySelectorAll('.find-similar-btn').forEach(btn => {
        btn.addEventListener('click', findSimilar);
    });
}

async function findSimilar(e) {
    const imageId = e.target.dataset.imageId;
    
    loading.classList.remove('hidden');
    error.classList.add('hidden');
    
    try {
        const response = await fetch('/api/search-similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId, top_k: 10 })
        });

        if (!response.ok) {
            throw new Error('Similar search failed');
        }

        const data = await response.json();
        displayResults(data.results);
        searchInput.value = `Similar to: ${imageId}`;
    } catch (err) {
        showError('Similar search failed: ' + err.message);
    } finally {
        loading.classList.remove('hidden');
    }
}

function showError(message) {
    error.textContent = message;
    error.classList.remove('hidden');
}
